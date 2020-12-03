import collections
import decimal
import functools
import numbers

import dask.dataframe as dd
import numpy as np
from dask.dataframe.groupby import SeriesGroupBy

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.pandas.core import numeric_types
from ibis.backends.pandas.execution.generic import execute_node


# TODO - aggregations - #2553
@execute_node.register(ops.Arbitrary, SeriesGroupBy, type(None))
def execute_arbitrary_series_groupby(op, data, _, aggcontext=None, **kwargs):
    how = op.how
    if how is None:
        how = 'first'

    if how not in {'first', 'last'}:
        raise com.OperationNotDefinedError(
            'Arbitrary {!r} is not supported'.format(how)
        )
    return aggcontext.agg(data, how)


@execute_node.register(ops.Negate, dd.Series)
def execute_series_negate(op, data, **kwargs):
    return data.mul(-1)


# TODO - grouping - #2553
@execute_node.register(ops.Negate, SeriesGroupBy)
def execute_series_group_by_negate(op, data, **kwargs):
    return execute_series_negate(op, data.obj, **kwargs).groupby(
        data.grouper.groupings
    )


def call_numpy_ufunc(func, op, data, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return data.apply(
            functools.partial(execute_node, op, **kwargs),
            meta=(data.name, "object"),
        )
    return func(data)


@execute_node.register(ops.UnaryOp, dd.Series)
def execute_series_unary_op(op, data, **kwargs):
    function = getattr(np, type(op).__name__.lower())
    return call_numpy_ufunc(function, op, data, **kwargs)


@execute_node.register((ops.Ceil, ops.Floor), dd.Series)
def execute_series_ceil(op, data, **kwargs):
    return_type = np.object_ if data.dtype == np.object_ else np.int64
    func = getattr(np, type(op).__name__.lower())
    return call_numpy_ufunc(func, op, data, **kwargs).astype(return_type)


def vectorize_object(op, arg, *args, **kwargs):
    # TODO - this works for now, but I think we can do something much better
    func = np.vectorize(functools.partial(execute_node, op, **kwargs))
    out = dd.from_array(func(arg, *args), columns=arg.name)
    out.index = arg.index
    return out


@execute_node.register(
    ops.Log, dd.Series, (dd.Series, numbers.Real, decimal.Decimal, type(None))
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


@execute_node.register(ops.Quantile, (dd.Series, SeriesGroupBy), numeric_types)
def execute_series_quantile(op, data, quantile, aggcontext=None, **kwargs):
    return data.quantile(q=quantile)


@execute_node.register(ops.MultiQuantile, dd.Series, collections.abc.Sequence)
def execute_series_quantile_sequence(
    op, data, quantile, aggcontext=None, **kwargs
):
    return list(data.quantile(q=quantile))


# TODO - aggregations - #2553
@execute_node.register(
    ops.MultiQuantile, SeriesGroupBy, collections.abc.Sequence
)
def execute_series_quantile_groupby(
    op, data, quantile, aggcontext=None, **kwargs
):
    def q(x, quantile, interpolation):
        result = x.quantile(quantile, interpolation=interpolation).tolist()
        res = [result for _ in range(len(x))]
        return res

    result = aggcontext.agg(data, q, quantile, op.interpolation)
    return result


@execute_node.register(
    ops.Round, dd.Series, (dd.Series, np.integer, type(None), int)
)
def execute_round_series(op, data, places, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, places, **kwargs)
    result = data.round(places or 0)
    return result if places else result.astype('int64')
