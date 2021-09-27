"""
Reduces sequences

NOTE: This file overwrite the pandas backend registered handlers for:

- execute_node_greatest_list,
- execute_node_least_list

This is so we can register our handlers that transparently handle both the'
dask specific types and pandas types. This cannot be done via the
dispatcher since the top level container is a list.
"""

import collections
import functools
from collections.abc import Sized

import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np
import toolz

import ibis.expr.operations as ops
from ibis.backends.pandas.execution.generic import (
    execute_node_greatest_list,
    execute_node_least_list,
)

from ..dispatch import execute_node
from .util import make_selected_obj


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


@execute_node.register(ops.Reduction, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, type(op).__name__.lower())


def _filtered_reduction(data, mask):
    return make_selected_obj(data)[mask.obj].groupby(data.index)


@execute_node.register(ops.Reduction, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_reduction_series_gb_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(grouped_and_filtered_data, type(op).__name__.lower())


@execute_node.register(ops.Reduction, dd.Series, (dd.Series, type(None)))
def execute_reduction_series_mask(op, data, mask, aggcontext=None, **kwargs):
    operand = data[mask] if mask is not None else data
    return aggcontext.agg(operand, type(op).__name__.lower())


@execute_node.register(
    (ops.CountDistinct, ops.HLLCardinality), ddgb.SeriesGroupBy, type(None)
)
def execute_count_distinct_series_groupby(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'nunique')


@execute_node.register(
    (ops.CountDistinct, ops.HLLCardinality),
    ddgb.SeriesGroupBy,
    ddgb.SeriesGroupBy,
)
def execute_count_distinct_series_groupby_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(grouped_and_filtered_data, "nunique")


@execute_node.register(
    (ops.CountDistinct, ops.HLLCardinality), dd.Series, (dd.Series, type(None))
)
def execute_count_distinct_series_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(data[mask] if mask is not None else data, 'nunique')


variance_ddof = {'pop': 0, 'sample': 1}


@execute_node.register(ops.Variance, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby_var(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'var', ddof=variance_ddof[op.how])


@execute_node.register(ops.Variance, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_var_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(
        grouped_and_filtered_data, "var", ddof=variance_ddof[op.how]
    )


@execute_node.register(ops.Variance, dd.Series, (dd.Series, type(None)))
def execute_variance_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        'var',
        ddof=variance_ddof[op.how],
    )


@execute_node.register(ops.StandardDev, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby_std(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'std', ddof=variance_ddof[op.how])


@execute_node.register(ops.StandardDev, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_std_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(
        grouped_and_filtered_data, "std", ddof=variance_ddof[op.how]
    )


@execute_node.register(ops.StandardDev, dd.Series, (dd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        'std',
        ddof=variance_ddof[op.how],
    )
