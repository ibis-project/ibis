from __future__ import annotations

import collections
import decimal
import functools
import numbers

import dask.dataframe as dd
import dask.dataframe.groupby as ddgb
import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution.util import make_selected_obj
from ibis.backends.pandas.core import numeric_types


@execute_node.register(ops.Negate, dd.Series)
def execute_series_negate(_, data, **kwargs):
    return -data


@execute_node.register(ops.Negate, ddgb.SeriesGroupBy)
def execute_series_group_by_negate(op, data, **kwargs):
    return execute_series_negate(op, make_selected_obj(data), **kwargs).groupby(
        data.index
    )


def call_numpy_ufunc(func, op, data, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return data.apply(
            functools.partial(execute_node, op, **kwargs),
            meta=(data.name, "object"),
        )
    return func(data)


@execute_node.register(ops.Unary, dd.Series)
def execute_series_unary_op(op, data, **kwargs):
    op_type = type(op)
    if op_type == ops.BitwiseNot:
        function = np.bitwise_not
    else:
        function = getattr(np, op_type.__name__.lower())
    return call_numpy_ufunc(function, op, data, **kwargs)


@execute_node.register(ops.Acos, dd.Series)
def execute_series_acos(_, data, **kwargs):
    return np.arccos(data)


@execute_node.register(ops.Asin, dd.Series)
def execute_series_asin(_, data, **kwargs):
    return np.arcsin(data)


@execute_node.register(ops.Atan, dd.Series)
def execute_series_atan(_, data, **kwargs):
    return np.arctan(data)


@execute_node.register(ops.Cot, dd.Series)
def execute_series_cot(_, data, **kwargs):
    return 1.0 / np.tan(data)


@execute_node.register(ops.Atan2, dd.Series, dd.Series)
@execute_node.register(ops.Atan2, numeric_types, dd.Series)
@execute_node.register(ops.Atan2, dd.Series, numeric_types)
def execute_series_atan2(_, y, x, **kwargs):
    return np.arctan2(y, x)


@execute_node.register((ops.Ceil, ops.Floor), dd.Series)
def execute_series_ceil(op, data, **kwargs):
    return_type = np.object_ if data.dtype == np.object_ else np.int64
    func = getattr(np, type(op).__name__.lower())
    return call_numpy_ufunc(func, op, data, **kwargs).astype(return_type)


def vectorize_object(op, arg, *args, **kwargs):
    # TODO - this works for now, but I think we can do something much better
    func = np.vectorize(functools.partial(execute_node, op, **kwargs))
    out = dd.from_array(func(arg, *args), columns=arg.name)
    return out


@execute_node.register(
    ops.Log,
    dd.Series,
    (dd.Series, pd.Series, numbers.Real, decimal.Decimal, type(None)),
)
def execute_series_log_with_base(op, data, base, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, base, **kwargs)

    if base is None:
        return np.log(data)
    return np.log(data) / np.log(base)


@execute_node.register(ops.Ln, dd.Series)
def execute_series_natural_log(op, data, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return data.apply(
            functools.partial(execute_node, op, **kwargs),
            meta=(data.name, "object"),
        )
    return np.log(data)


@execute_node.register(ops.Quantile, dd.Series, numeric_types, (dd.Series, type(None)))
def execute_series_quantile(op, data, quantile, mask, **_):
    if mask is not None:
        data = data.loc[mask]
    return data.quantile(q=quantile)


@execute_node.register(ops.Quantile, ddgb.SeriesGroupBy, numeric_types, type(None))
def execute_series_quantile_group_by(op, data, quantile, mask, **_):
    raise NotImplementedError(
        "Quantile not implemented for Dask SeriesGroupBy, Dask #9824"
    )


@execute_node.register(
    ops.MultiQuantile, dd.Series, collections.abc.Sequence, type(None)
)
def execute_series_quantile_sequence(op, data, quantile, mask, **_):
    return list(data.quantile(q=quantile))


# TODO - aggregations - #2553
@execute_node.register(
    ops.MultiQuantile, ddgb.SeriesGroupBy, collections.abc.Sequence, type(None)
)
def execute_series_quantile_groupby(op, data, quantile, mask, aggcontext=None, **_):
    def q(x, quantile):
        result = x.quantile(quantile).tolist()
        return [result for _ in range(len(x))]

    return aggcontext.agg(data, q, quantile)


@execute_node.register(ops.Round, dd.Series, (dd.Series, np.integer, type(None), int))
def execute_round_series(op, data, places, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, places, **kwargs)
    result = data.round(places or 0)
    return result if places else result.astype("int64")
