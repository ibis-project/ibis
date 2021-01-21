"""
Reduces sequences

NOTE: This file overwrite the pandas backend registered handlers for:

- execute_node_expr_list,
- execute_node_greatest_list,
- execute_node_least_list

This is so we can register our handlers that transparently handle both the'
dask specific types and pandas types. This cannot be done via the
dispatcher since the top level container is a list.
"""

import collections
import functools
import operator
from collections.abc import Sized

import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np
import pandas as pd
import toolz

import ibis
import ibis.expr.operations as ops
from ibis.backends.pandas.core import execute
from ibis.backends.pandas.execution.generic import (
    execute_node,
    execute_node_expr_list,
    execute_node_greatest_list,
    execute_node_least_list,
)


@toolz.curry
def promote_to_sequence(length, obj):
    if isinstance(obj, dd.Series):
        # we must force length computation if we have mixed types
        # otherwise da.reductions can't compare arrays
        return obj.to_dask_array(lengths=True)
    else:
        return da.from_array(np.repeat(obj, length))


def pairwise_reducer(func, values):
    return functools.reduce(lambda x, y: func(x, y), values)


def compute_row_reduction(func, value):
    final_sizes = {len(x) for x in value if isinstance(x, Sized)}
    if not final_sizes:
        return func(value)
    (final_size,) = final_sizes
    arrays = list(map(promote_to_sequence(final_size), value))
    raw = pairwise_reducer(func, arrays)
    return dd.from_array(raw).squeeze()


@execute_node.register(ops.Greatest, collections.abc.Sequence)
def dask_execute_node_greatest_list(op, value, **kwargs):
    if all(type(v) != dd.Series for v in value):
        return execute_node_greatest_list(op, value, **kwargs)
    return compute_row_reduction(da.maximum, value)


@execute_node.register(ops.Least, collections.abc.Sequence)
def dask_execute_node_least_list(op, value, **kwargs):
    if all(type(v) != dd.Series for v in value):
        return execute_node_least_list(op, value, **kwargs)
    return compute_row_reduction(da.minimum, value)


# TODO - aggregations - #2553
@execute_node.register(ops.Reduction, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, type(op).__name__.lower())


def _filtered_reduction(mask, method, data):
    return method(data[mask[data.index]])


# TODO - aggregations - #2553
@execute_node.register(ops.Reduction, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_reduction_series_gb_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    method = operator.methodcaller(type(op).__name__.lower())
    return aggcontext.agg(
        data, functools.partial(_filtered_reduction, mask.obj, method)
    )


# TODO - aggregations - #2553
@execute_node.register(ops.Reduction, dd.Series, (dd.Series, type(None)))
def execute_reduction_series_mask(op, data, mask, aggcontext=None, **kwargs):
    operand = data[mask] if mask is not None else data
    return aggcontext.agg(operand, type(op).__name__.lower())


# TODO - aggregations - #2553
@execute_node.register(
    (ops.CountDistinct, ops.HLLCardinality), ddgb.SeriesGroupBy, type(None)
)
def execute_count_distinct_series_groupby(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'nunique')


# TODO - aggregations - #2553
@execute_node.register(
    (ops.CountDistinct, ops.HLLCardinality),
    ddgb.SeriesGroupBy,
    ddgb.SeriesGroupBy,
)
def execute_count_distinct_series_groupby_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(
        data,
        functools.partial(_filtered_reduction, mask.obj, dd.Series.nunique),
    )


# TODO - aggregations - #2553
@execute_node.register(
    (ops.CountDistinct, ops.HLLCardinality), dd.Series, (dd.Series, type(None))
)
def execute_count_distinct_series_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(data[mask] if mask is not None else data, 'nunique')


variance_ddof = {'pop': 0, 'sample': 1}


# TODO - aggregations - #2553
@execute_node.register(ops.Variance, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby_var(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'var', ddof=variance_ddof[op.how])


# TODO - aggregations - #2553
@execute_node.register(ops.Variance, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_var_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data,
        lambda x, mask=mask.obj, ddof=variance_ddof[op.how]: (
            x[mask[x.index]].var(ddof=ddof)
        ),
    )


# TODO - aggregations - #2553
@execute_node.register(ops.Variance, dd.Series, (dd.Series, type(None)))
def execute_variance_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        'var',
        ddof=variance_ddof[op.how],
    )


# TODO - aggregations - #2553
@execute_node.register(ops.StandardDev, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby_std(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'std', ddof=variance_ddof[op.how])


# TODO - aggregations - #2553
@execute_node.register(ops.StandardDev, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_std_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data,
        lambda x, mask=mask.obj, ddof=variance_ddof[op.how]: (
            x[mask[x.index]].std(ddof=ddof)
        ),
    )


# TODO - aggregations - #2553
@execute_node.register(ops.StandardDev, dd.Series, (dd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        'std',
        ddof=variance_ddof[op.how],
    )


@execute_node.register(ops.ExpressionList, collections.abc.Sequence)
def dask_execute_node_expr_list(op, sequence, **kwargs):
    if all(type(s) != dd.Series for s in sequence):
        execute_node_expr_list(op, sequence, **kwargs)
    columns = [e.get_name() for e in op.exprs]
    schema = ibis.schema(list(zip(columns, (e.type() for e in op.exprs))))
    data = {col: [execute(el, **kwargs)] for col, el in zip(columns, sequence)}
    return schema.apply_to(
        dd.from_pandas(pd.DataFrame(data, columns=columns), npartitions=1)
    )
