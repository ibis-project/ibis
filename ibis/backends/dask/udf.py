from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import dask.delayed
import pandas as pd

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend
from ibis.backends.dask.aggcontext import Transform
from ibis.backends.dask.dispatch import execute_node, pre_execute
from ibis.backends.dask.execution.util import (
    assert_identical_grouping_keys,
    make_meta_series,
    make_selected_obj,
)
from ibis.backends.pandas.udf import create_gens_from_args_groupby

if TYPE_CHECKING:
    import numpy as np


def make_struct_op_meta(op: ir.Expr) -> list[tuple[str, np.dtype]]:
    """Unpacks a dt.Struct into a DataFrame meta."""
    return list(
        zip(
            op.return_type.names,
            [x.to_pandas() for x in op.return_type.types],
        )
    )


@pre_execute.register(ops.ElementWiseVectorizedUDF)
@pre_execute.register(ops.ElementWiseVectorizedUDF, BaseBackend)
def pre_execute_elementwise_udf(op, *clients, scope=None, **kwargs):
    """Register execution rules for elementwise UDFs."""
    input_type = op.input_type

    # definitions

    # Define an execution rule for elementwise operations on a
    # grouped Series
    nargs = len(input_type)

    @execute_node.register(
        ops.ElementWiseVectorizedUDF,
        *(itertools.repeat(ddgb.SeriesGroupBy, nargs)),
    )
    def execute_udf_node_groupby(op, *args, **kwargs):
        func = op.func

        # all grouping keys must be identical
        assert_identical_grouping_keys(*args)

        # we're performing a scalar operation on grouped column, so
        # perform the operation directly on the underlying Series
        # and regroup after it's finished
        args_objs = [make_selected_obj(arg) for arg in args]
        groupings = args[0].index
        return dd.map_partitions(func, *args_objs).groupby(groupings)

    # Define an execution rule for a simple elementwise Series
    # function
    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(dd.Series, nargs))
    )
    def execute_udf_node(op, *args, cache=None, timecontext=None, **kwargs):
        # We have rewritten op.func to be a closure enclosing
        # the kwargs, and therefore, we do not need to pass
        # kwargs here. This is true for all udf execution in this
        # file.
        # See ibis.legacy.udf.vectorized.UserDefinedFunction
        with contextlib.suppress(KeyError):
            return cache[(op, timecontext)]

        if op.return_type.is_struct():
            meta = make_struct_op_meta(op)
            df = dd.map_partitions(op.func, *args, meta=meta)
        else:
            name = args[0].name if len(args) == 1 else None
            meta = pd.Series([], name=name, dtype=op.return_type.to_pandas())
            df = dd.map_partitions(op.func, *args, meta=meta)

        cache[(op, timecontext)] = df

        return df

    return scope


@pre_execute.register(ops.AnalyticVectorizedUDF)
@pre_execute.register(ops.AnalyticVectorizedUDF, BaseBackend)
@pre_execute.register(ops.ReductionVectorizedUDF)
@pre_execute.register(ops.ReductionVectorizedUDF, BaseBackend)
def pre_execute_analytic_and_reduction_udf(op, *clients, scope=None, **kwargs):
    input_type = op.input_type
    nargs = len(input_type)

    # An execution rule to handle analytic and reduction UDFs over
    # 1) an ungrouped window,
    # 2) an ungrouped Aggregate node, or
    # 3) an ungrouped custom aggregation context
    # Ungrouped analytic/reduction functions receive the entire Series at once
    # This is generally not recommended.
    @execute_node.register(type(op), *(itertools.repeat(dd.Series, nargs)))
    def execute_udaf_node_no_groupby(op, *args, aggcontext, **kwargs):
        # This function is in essence fully materializing the dd.Series and
        # passing that (now) pd.Series to aggctx. This materialization
        # happens at `.compute()` time, making this "lazy"
        @dask.delayed
        def lazy_agg(*series: pd.Series):
            return aggcontext.agg(series[0], op.func, *series[1:])

        lazy_result = lazy_agg(*args)

        # Depending on the type of operation, lazy_result is a Delayed that
        # could become a dd.Series or a dd.core.Scalar
        if isinstance(op, ops.AnalyticVectorizedUDF):
            if op.return_type.is_struct():
                meta = make_struct_op_meta(op)
            else:
                meta = make_meta_series(
                    dtype=op.return_type.to_pandas(),
                    name=args[0].name,
                )
            result = dd.from_delayed(lazy_result, meta=meta)

            if args[0].known_divisions:
                if not len({a.divisions for a in args}) == 1:
                    raise ValueError("Mixed divisions passed to AnalyticVectorized UDF")
                # result is going to be a single partitioned thing, but we
                # need it to be able to dd.concat it with other data
                # downstream. We know that this udf operation did not change
                # the index. Thus, we know the divisions, allowing dd.concat
                # to align this piece with the other pieces.
                original_divisions = args[0].divisions
                result.divisions = (
                    original_divisions[0],
                    original_divisions[-1],
                )
                result = result.repartition(divisions=original_divisions)
        else:
            # lazy_result is a dd.core.Scalar from an ungrouped reduction
            return_type = op.return_type
            if return_type.is_array() or return_type.is_struct():
                # we're outputting a dt.Struct that will need to be destructured
                # or an array of an unknown size.
                # we compute so we can work with items inside downstream.
                result = lazy_result.compute()
            else:
                # manually construct a dd.core.Scalar out of the delayed result
                result = dd.from_delayed(
                    lazy_result,
                    meta=op.return_type.to_pandas(),
                    # otherwise dask complains this is a scalar
                    verify_meta=False,
                )

        return result

    @execute_node.register(
        ops.ReductionVectorizedUDF,
        *(itertools.repeat(ddgb.SeriesGroupBy, nargs)),
    )
    def execute_reduction_node_groupby(op, *args, aggcontext, **kwargs):
        # To apply a udf func to a list of grouped series we:
        # 1. Grab the dataframe they're grouped off of
        # 2. Grab the column name for each series
        # 3. .apply a wrapper that performs the selection using the col name
        #    and applies the udf on to those
        # This way we rely on dask dealing with groups and pass the udf down
        # to the frame level.
        assert_identical_grouping_keys(*args)

        func = op.func
        groupings = args[0].index
        parent_df = args[0].obj

        if isinstance(aggcontext, Transform):
            # We are aggregating over an unbounded (and GROUPED) window,
            # which uses a Transform aggregation context.
            # We need to do some pre-processing to func and args so that
            # Transform can pull data out of the SeriesGroupBys in args.

            # Construct a generator that yields the next group of data
            # for every argument excluding the first (pandas performs
            # the iteration for the first argument) for each argument
            # that is a SeriesGroupBy.
            iters = create_gens_from_args_groupby(*args[1:])

            # TODO: Unify calling convention here to be more like
            # window
            def aggregator(first, *rest):
                # map(next, *rest) gets the inputs for the next group
                # TODO: might be inefficient to do this on every call
                return func(first, *map(next, rest))

            return aggcontext.agg(args[0], aggregator, *iters)
        else:
            columns = [parent_df[idx] for idx in args[0].index]
            for arg in args:
                df = arg.obj
                column = df[arg._meta.obj.name]
                columns.append(column)
            parent_df = dd.concat(columns, axis=1)

            out_type = op.return_type.to_pandas()

            grouped_df = parent_df.groupby(groupings)
            col_names = [col._meta._selected_obj.name for col in args]

            def apply_wrapper(df, apply_func, col_names):
                cols = (df[col] for col in col_names)
                return apply_func(*cols)

            if len(groupings) > 1:
                meta_index = pd.MultiIndex.from_arrays(
                    [[0]] * len(groupings), names=groupings
                )
                meta_value = [dd.utils.make_meta(out_type)]
            else:
                meta_index = pd.Index([], name=groupings[0])
                meta_value = []

            return grouped_df.apply(
                apply_wrapper,
                func,
                col_names,
                meta=pd.Series(meta_value, index=meta_index, dtype=out_type),
            )

    @execute_node.register(
        ops.AnalyticVectorizedUDF,
        *(itertools.repeat(ddgb.SeriesGroupBy, nargs)),
    )
    def execute_analytic_node_groupby(op, *args, aggcontext, **kwargs):
        # To apply a udf func to a list of grouped series we:
        # 1. Grab the dataframe they're grouped off of
        # 2. Grab the column name for each series
        # 3. .apply a wrapper that performs the selection using the col name
        #    and applies the udf on to those
        # This way we rely on dask dealing with groups and pass the udf down
        # to the frame level.
        assert_identical_grouping_keys(*args)

        func = op.func
        groupings = args[0].index
        parent_df = args[0].obj

        columns = [parent_df[idx] for idx in groupings]
        columns.extend(arg.obj[arg._meta.obj.name] for arg in args)
        parent_df = dd.concat(columns, axis=1)

        out_type = op.return_type.to_pandas()

        grouped_df = parent_df.groupby(groupings)
        col_names = [col._meta._selected_obj.name for col in args]

        def apply_wrapper(df, apply_func, col_names):
            cols = (df[col] for col in col_names)
            return apply_func(*cols)

        if op.return_type.is_struct():
            # with struct output we destruct to a dataframe directly
            meta = dd.utils.make_meta(make_struct_op_meta(op))
            meta.index.name = parent_df.index.name
            result = grouped_df.apply(
                apply_wrapper,
                func,
                col_names,
                meta=meta,
            )
            # we don't know how data moved around here
            result = result.reset_index().set_index(parent_df.index.name)
        else:
            # after application we will get a series with a multi-index of
            # groupings + index
            meta_index = pd.MultiIndex.from_arrays(
                [[0]] * (len(groupings) + 1),
                names=groupings + [parent_df.index.name],
            )
            meta_value = [dd.utils.make_meta(out_type)]

            result = grouped_df.apply(
                apply_wrapper,
                func,
                col_names,
                meta=pd.Series(meta_value, index=meta_index, dtype=out_type),
            )

        return result

    return scope
