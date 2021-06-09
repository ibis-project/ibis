import itertools

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import dask.delayed
import pandas

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base import Client
from ibis.backends.pandas.udf import nullable  # noqa

from .dispatch import execute_node, pre_execute
from .execution.util import (
    assert_identical_grouping_keys,
    make_selected_obj,
    safe_scalar_type,
)


@pre_execute.register(ops.ElementWiseVectorizedUDF)
@pre_execute.register(ops.ElementWiseVectorizedUDF, Client)
def pre_execute_elementwise_udf(op, *clients, scope=None, **kwargs):
    """Register execution rules for elementwise UDFs.
    """
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
        if isinstance(op._output_type, dt.Struct):
            meta = list(
                zip(
                    op._output_type.names,
                    [x.to_dask() for x in op._output_type.types],
                )
            )

            df = dd.map_partitions(op.func, *args, meta=meta)
            return df
        else:
            name = args[0].name if len(args) == 1 else None
            meta = pandas.Series(
                [], name=name, dtype=op._output_type.to_dask()
            )
            df = dd.map_partitions(op.func, *args, meta=meta)

            return df

    @execute_node.register(
        ops.ElementWiseVectorizedUDF, *(itertools.repeat(object, nargs))
    )
    def execute_udf_node_non_dask(op, *args, **kwargs):
        return op.func(*args)

    return scope


@pre_execute.register(ops.AnalyticVectorizedUDF)
@pre_execute.register(ops.AnalyticVectorizedUDF, Client)
@pre_execute.register(ops.ReductionVectorizedUDF)
@pre_execute.register(ops.ReductionVectorizedUDF, Client)
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
            meta = pandas.Series(
                [], name=args[0].name, dtype=op._output_type.to_dask()
            )
            result = dd.from_delayed(lazy_result, meta=meta)
        else:
            # lazy_result is a dd.core.Scalar from an ungrouped reduction
            if isinstance(op._output_type, (dt.Array, dt.Struct)):
                # we're outputing a dt.Struct that will need to be destructured
                # or an array of an unknown size.
                # we compute so we can work with items inside downstream.
                result = lazy_result.compute()
            else:
                output_meta = safe_scalar_type(op._output_type.to_dask())
                result = dd.from_delayed(
                    lazy_result, meta=output_meta, verify_meta=False
                )

        return result

    @execute_node.register(
        type(op), *(itertools.repeat(ddgb.SeriesGroupBy, nargs))
    )
    def execute_udaf_node_groupby(op, *args, aggcontext, **kwargs):
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
        out_type = op._output_type.to_dask()

        grouped_df = args[0].obj.groupby(groupings)
        col_names = [col._meta._selected_obj.name for col in args]

        def apply_wrapper(df, apply_func, col_names):
            cols = (df[col] for col in col_names)
            return apply_func(*cols)

        if len(groupings) > 1:
            meta_index = pandas.MultiIndex.from_arrays(
                [[0], [0]], names=groupings
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

    return scope
