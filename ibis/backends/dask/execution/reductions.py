"""Reduces sequences.

NOTE: This file overwrite the pandas backend registered handlers for:

- execute_node_greatest_list,
- execute_node_least_list

This is so we can register our handlers that transparently handle both the'
dask specific types and pandas types. This cannot be done via the
dispatcher since the top level container is a list.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Sized

import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np
import toolz
from multipledispatch.variadic import Variadic

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import make_selected_obj
from ibis.backends.pandas.execution.generic import (
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


def compute_row_reduction(func, values):
    final_sizes = {len(x) for x in values if isinstance(x, Sized)}
    if not final_sizes:
        return func(values)
    (final_size,) = final_sizes
    arrays = list(map(promote_to_sequence(final_size), values))
    raw = pairwise_reducer(func, arrays)
    return dd.from_array(raw).squeeze()


# XXX: there's non-determinism in the dask and pandas dispatch registration of
# Greatest/Least/Coalesce, because 1) dask and pandas share `execute_node`
# which is a design flaw and 2) greatest/least/coalesce need to handle
# mixed-type (the Series types plus any related scalar type) inputs so `object`
# is used as a possible input type.
#
# Here we remove the dispatch for pandas if it exists because the dask rule
# handles both cases.
with contextlib.suppress(KeyError):
    del execute_node[ops.Greatest, Variadic[object]]


with contextlib.suppress(KeyError):
    del execute_node[ops.Least, Variadic[object]]


@execute_node.register(ops.Greatest, [(object, dd.Series)])
def dask_execute_node_greatest_list(op, *values, **kwargs):
    if all(type(v) != dd.Series for v in values):
        return execute_node_greatest_list(op, *values, **kwargs)
    return compute_row_reduction(da.maximum, values)


@execute_node.register(ops.Least, [(object, dd.Series)])
def dask_execute_node_least_list(op, *values, **kwargs):
    if all(type(v) != dd.Series for v in values):
        return execute_node_least_list(op, *values, **kwargs)
    return compute_row_reduction(da.minimum, values)


@execute_node.register(ops.Reduction, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(data, type(op).__name__.lower())


def _filtered_reduction(data, mask):
    return make_selected_obj(data)[mask.obj].groupby(data.index)


@execute_node.register(ops.Reduction, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_reduction_series_gb_mask(op, data, mask, aggcontext=None, **kwargs):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(grouped_and_filtered_data, type(op).__name__.lower())


@execute_node.register(ops.Reduction, dd.Series, (dd.Series, type(None)))
def execute_reduction_series_mask(op, data, mask, aggcontext=None, **kwargs):
    operand = data[mask] if mask is not None else data
    return aggcontext.agg(operand, type(op).__name__.lower())


@execute_node.register(
    (ops.First, ops.Last), ddgb.SeriesGroupBy, (ddgb.SeriesGroupBy, type(None))
)
@execute_node.register((ops.First, ops.Last), dd.Series, (dd.Series, type(None)))
def execute_first_last_dask(op, data, mask, aggcontext=None, **kwargs):
    raise exc.OperationNotDefinedError(
        "Dask does not support first or last aggregations"
    )


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    ddgb.SeriesGroupBy,
    type(None),
)
def execute_count_distinct_series_groupby(op, data, _, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "nunique")


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    ddgb.SeriesGroupBy,
    ddgb.SeriesGroupBy,
)
def execute_count_distinct_series_groupby_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(grouped_and_filtered_data, "nunique")


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    dd.Series,
    (dd.Series, type(None)),
)
def execute_count_distinct_series_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(data[mask] if mask is not None else data, "nunique")


variance_ddof = {"pop": 0, "sample": 1}


@execute_node.register(ops.Variance, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby_var(op, data, _, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "var", ddof=variance_ddof[op.how])


@execute_node.register(ops.Variance, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_var_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(grouped_and_filtered_data, "var", ddof=variance_ddof[op.how])


@execute_node.register(ops.Variance, dd.Series, (dd.Series, type(None)))
def execute_variance_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        "var",
        ddof=variance_ddof[op.how],
    )


@execute_node.register(ops.StandardDev, ddgb.SeriesGroupBy, type(None))
def execute_reduction_series_groupby_std(op, data, _, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "std", ddof=variance_ddof[op.how])


@execute_node.register(ops.StandardDev, ddgb.SeriesGroupBy, ddgb.SeriesGroupBy)
def execute_std_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    grouped_and_filtered_data = _filtered_reduction(data, mask)
    return aggcontext.agg(grouped_and_filtered_data, "std", ddof=variance_ddof[op.how])


@execute_node.register(ops.StandardDev, dd.Series, (dd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        "std",
        ddof=variance_ddof[op.how],
    )


@execute_node.register(ops.ArgMax, dd.Series, dd.Series, (dd.Series, type(None)))
def execute_argmax_series(op, data, key, mask, aggcontext=None, **kwargs):
    idxmax = aggcontext.agg(key[mask] if mask is not None else key, "idxmax").compute()
    return data.loc[idxmax]


@execute_node.register(ops.ArgMin, dd.Series, dd.Series, (dd.Series, type(None)))
def execute_argmin_series(op, data, key, mask, aggcontext=None, **kwargs):
    idxmin = aggcontext.agg(key[mask] if mask is not None else key, "idxmin").compute()
    return data.loc[idxmin]
