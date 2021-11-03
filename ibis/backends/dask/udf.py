import itertools
from typing import List, Tuple

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import dask.delayed
import numpy as np
import pandas

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend
from ibis.backends.pandas.udf import nullable  # noqa

from .dispatch import execute_node, pre_execute
from .execution.util import (
    assert_identical_grouping_keys,
    make_meta_series,
    make_selected_obj,
    safe_scalar_type,
)


def make_struct_op_meta(op: ir.Expr) -> List[Tuple[str, np.dtype]]:
    """Unpacks a dt.Struct into a DataFrame meta"""
    return list(
        zip(
            op.return_type.names,
            [x.to_dask() for x in op.return_type.types],
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
    def execute_udf_node(op, *args, **kwargs):
        # We have rewritten op.func to be a closure enclosing
        # the kwargs, and therefore, we do not need to pass
        # kwargs here. This is true for all udf execution in this
        # file.
        # See ibis.udf.vectorized.UserDefinedFunction
        if isinstance(op.return_type, dt.Struct):
            meta = make_struct_op_meta(op)

            df = dd.map_partitions(op.func, *args, meta=meta)
            return df
        else:
            name = args[0].name if len(args) == 1 else None
            meta = pandas.Series([], name=name, dtype=op.return_type.to_dask())
            df = dd.map_partitions(op.func, *args, meta=meta)

            return df

    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(object, nargs))
    )
    def execute_udf_node_non_dask(op, *args, **kwargs):
        return op.func(*args)

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
    # Ungrouped analytic/reduction functions recieve the entire Series at once
    # This is generally not recommened.
    @execute_node.register(type(op), *(itertools.repeat(dd.Series, nargs)))
    def execute_udaf_node_no_groupby(op, *args, aggcontext, **kwargs):

        # This function is in essence fully materializing the dd.Series and
        # passing that (now) pd.Series to aggctx. This materialization
        # happens at `.compute()` time, making this "lazy"
        @dask.delayed
        def lazy_agg(*series: pandas.Series):
            return aggcontext.agg(series[0], op.func, *series[1:])

        lazy_result = lazy_agg(*args)

        # Depending on the type of operation, lazy_result is a Delayed that
        # could become a dd.Series or a dd.core.Scalar
        if isinstance(op, ops.AnalyticVectorizedUDF):
            if isinstance(op.return_type, dt.Struct):
                meta = make_struct_op_meta(op)
            else:
                meta = make_meta_series(
                    dtype=op.return_type.to_dask(),
                    name=args[0].name,
                )
            result = dd.from_delayed(lazy_result, meta=meta)

            if args[0].known_divisions:
                if not len({a.divisions for a in args}) == 1:
                    raise ValueError(
                        "Mixed divisions passed to AnalyticVectorized UDF"
                    )
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
            if isinstance(op.return_type, (dt.Array, dt.Struct)):
                # we're outputing a dt.Struct that will need to be destructured
                # or an array of an unknown size.
                # we compute so we can work with items inside downstream.
                result = lazy_result.compute()
            else:
                output_meta = safe_scalar_type(op.return_type.to_dask())
                result = dd.from_delayed(
                    lazy_result, meta=output_meta, verify_meta=False
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
        out_type = op.return_type.to_dask()

        grouped_df = parent_df.groupby(groupings)
        col_names = [col._meta._selected_obj.name for col in args]

        def apply_wrapper(df, apply_func, col_names):
            cols = (df[col] for col in col_names)
            return apply_func(*cols)

        if len(groupings) > 1:
            meta_index = pandas.MultiIndex.from_arrays(
                [[0]] * len(groupings), names=groupings
            )
            meta_value = [dd.utils.make_meta(safe_scalar_type(out_type))]
        else:
            meta_index = pandas.Index([], name=groupings[0])
            meta_value = []

        return grouped_df.apply(
            apply_wrapper,
            func,
            col_names,
            meta=pandas.Series(meta_value, index=meta_index, dtype=out_type),
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
        out_type = op.return_type.to_dask()

        grouped_df = parent_df.groupby(groupings)
        col_names = [col._meta._selected_obj.name for col in args]

        def apply_wrapper(df, apply_func, col_names):
            cols = (df[col] for col in col_names)
            return apply_func(*cols)

        if isinstance(op.return_type, dt.Struct):
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
            meta_index = pandas.MultiIndex.from_arrays(
                [[0]] * (len(groupings) + 1),
                names=groupings + [parent_df.index.name],
            )
            meta_value = [dd.utils.make_meta(safe_scalar_type(out_type))]

            result = grouped_df.apply(
                apply_wrapper,
                func,
                col_names,
                meta=pandas.Series(
                    meta_value, index=meta_index, dtype=out_type
                ),
            )
            # If you use the UDF directly (as in `test_udaf_analytic_groupby`)
            # we need to do some renaming/cleanup to get the result to
            # conform to what the pandas output would look like. Nomrally this
            # is handled in `util.coerce_to_output`.
            parent_idx_name = parent_df.index.name
            # These defaults are chosen by pandas
            result_idx_name = parent_idx_name if parent_idx_name else "level_1"
            result_value_name = result.name if result.name else 0
            # align the results back to the parent frame
            result = result.reset_index().set_index(result_idx_name)
            result = result[result_value_name]

        return result

    return scope
