"""Execution rules for generic ibis operations."""

import collections
import datetime
import decimal
import functools
import math
import numbers
import operator
from collections.abc import Sized
from typing import Dict, Optional

import numpy as np
import pandas as pd
import toolz
from pandas.api.types import DatetimeTZDtype
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.pandas import Backend as PandasBackend
from ibis.backends.pandas import aggcontext as agg_ctx
from ibis.backends.pandas.client import PandasTable
from ibis.backends.pandas.core import (
    boolean_types,
    date_types,
    execute,
    fixed_width_types,
    floating_types,
    integer_types,
    numeric_types,
    scalar_types,
    simple_types,
    timedelta_types,
    timestamp_types,
)
from ibis.backends.pandas.dispatch import execute_literal, execute_node
from ibis.backends.pandas.execution import constants
from ibis.backends.pandas.execution.util import coerce_to_output
from ibis.expr.scope import Scope
from ibis.expr.timecontext import get_time_col
from ibis.expr.typing import TimeContext


# By default return the literal value
@execute_literal.register(ops.Literal, object, dt.DataType)
def execute_node_literal_value_datatype(op, value, datatype, **kwargs):
    return value


# Because True and 1 hash to the same value, if we have True or False in scope
# keys while executing anything that should evaluate to 1 or 0 evaluates to
# True or False respectively. This is a hack to work around that by casting the
# bool to an integer.
@execute_literal.register(ops.Literal, object, dt.Integer)
def execute_node_literal_any_integer_datatype(op, value, datatype, **kwargs):
    return int(value)


@execute_literal.register(ops.Literal, object, dt.Boolean)
def execute_node_literal_any_boolean_datatype(op, value, datatype, **kwargs):
    return bool(value)


@execute_literal.register(ops.Literal, object, dt.Floating)
def execute_node_literal_any_floating_datatype(op, value, datatype, **kwargs):
    return float(value)


@execute_literal.register(ops.Literal, object, dt.Array)
def execute_node_literal_any_array_datatype(op, value, datatype, **kwargs):
    return np.array(value)


@execute_literal.register(ops.Literal, dt.DataType)
def execute_node_literal_datatype(op, datatype, **kwargs):
    return op.value


@execute_literal.register(
    ops.Literal, timedelta_types + (str,) + integer_types, dt.Interval
)
def execute_interval_literal(op, value, dtype, **kwargs):
    return pd.Timedelta(value, dtype.unit)


@execute_node.register(ops.Limit, pd.DataFrame, integer_types, integer_types)
def execute_limit_frame(op, data, nrows, offset, **kwargs):
    return data.iloc[offset : offset + nrows]


@execute_node.register(ops.Cast, SeriesGroupBy, dt.DataType)
def execute_cast_series_group_by(op, data, type, **kwargs):
    result = execute_cast_series_generic(op, data.obj, type, **kwargs)
    return result.groupby(data.grouper.groupings)


@execute_node.register(ops.Cast, pd.Series, dt.DataType)
def execute_cast_series_generic(op, data, type, **kwargs):
    return data.astype(constants.IBIS_TYPE_TO_PANDAS_TYPE[type])


@execute_node.register(ops.Cast, pd.Series, dt.Array)
def execute_cast_series_array(op, data, type, **kwargs):
    value_type = type.value_type
    numpy_type = constants.IBIS_TYPE_TO_PANDAS_TYPE.get(value_type, None)
    if numpy_type is None:
        raise ValueError(
            'Array value type must be a primitive type '
            '(e.g., number, string, or timestamp)'
        )
    return data.map(
        lambda array, numpy_type=numpy_type: array.astype(numpy_type)
    )


@execute_node.register(ops.Cast, pd.Series, dt.Timestamp)
def execute_cast_series_timestamp(op, data, type, **kwargs):
    arg = op.arg
    from_type = arg.type()

    if from_type.equals(type):  # noop cast
        return data

    tz = type.timezone

    if isinstance(from_type, (dt.Timestamp, dt.Date)):
        return data.astype(
            'M8[ns]' if tz is None else DatetimeTZDtype('ns', tz)
        )

    if isinstance(from_type, (dt.String, dt.Integer)):
        timestamps = pd.to_datetime(data.values, infer_datetime_format=True)
        if getattr(timestamps.dtype, "tz", None) is not None:
            method_name = "tz_convert"
        else:
            method_name = "tz_localize"
        method = getattr(timestamps, method_name)
        timestamps = method(tz)
        return pd.Series(timestamps, index=data.index, name=data.name)

    raise TypeError(f"Don't know how to cast {from_type} to {type}")


def _normalize(values, original_index, name, timezone=None):
    index = pd.DatetimeIndex(values, tz=timezone)
    return pd.Series(index.normalize(), index=original_index, name=name)


@execute_node.register(ops.Cast, pd.Series, dt.Date)
def execute_cast_series_date(op, data, type, **kwargs):
    arg = op.args[0]
    from_type = arg.type()

    if from_type.equals(type):
        return data

    if isinstance(from_type, dt.Timestamp):
        return _normalize(
            data.values, data.index, data.name, timezone=from_type.timezone
        )

    if from_type.equals(dt.string):
        values = data.values
        datetimes = pd.to_datetime(values, infer_datetime_format=True)
        try:
            datetimes = datetimes.tz_convert(None)
        except TypeError:
            pass
        dates = _normalize(datetimes, data.index, data.name)
        return pd.Series(dates, index=data.index, name=data.name)

    if isinstance(from_type, dt.Integer):
        return pd.Series(
            pd.to_datetime(data.values, unit='D').values,
            index=data.index,
            name=data.name,
        )

    raise TypeError(f"Don't know how to cast {from_type} to {type}")


@execute_node.register(ops.SortKey, pd.Series, bool)
def execute_sort_key_series_bool(op, data, ascending, **kwargs):
    return data


def call_numpy_ufunc(func, op, data, **kwargs):
    if getattr(data, "dtype", None) == np.dtype(np.object_):
        return data.apply(functools.partial(execute_node, op, **kwargs))
    return func(data)


@execute_node.register(ops.Negate, fixed_width_types + timedelta_types)
def execute_obj_negate(op, data, **kwargs):
    return -data


@execute_node.register(ops.Negate, pd.Series)
def execute_series_negate(op, data, **kwargs):
    return call_numpy_ufunc(np.negative, op, data, **kwargs)


@execute_node.register(ops.Negate, SeriesGroupBy)
def execute_series_group_by_negate(op, data, **kwargs):
    return execute_series_negate(op, data.obj, **kwargs).groupby(
        data.grouper.groupings
    )


@execute_node.register(ops.Unary, pd.Series)
def execute_series_unary_op(op, data, **kwargs):
    function = getattr(np, type(op).__name__.lower())
    return call_numpy_ufunc(function, op, data, **kwargs)


@execute_node.register(ops.Acos, (pd.Series, *numeric_types))
def execute_series_acos(_, data, **kwargs):
    return np.arccos(data)


@execute_node.register(ops.Asin, (pd.Series, *numeric_types))
def execute_series_asin(_, data, **kwargs):
    return np.arcsin(data)


@execute_node.register(ops.Atan, (pd.Series, *numeric_types))
def execute_series_atan(_, data, **kwargs):
    return np.arctan(data)


@execute_node.register(ops.Cot, (pd.Series, *numeric_types))
def execute_series_cot(_, data, **kwargs):
    return np.cos(data) / np.sin(data)


@execute_node.register(
    ops.Atan2, (pd.Series, *numeric_types), (pd.Series, *numeric_types)
)
def execute_series_atan2(_, y, x, **kwargs):
    return np.arctan2(y, x)


@execute_node.register(
    (ops.Cos, ops.Sin, ops.Tan),
    (pd.Series, *numeric_types),
)
def execute_series_trig(op, data, **kwargs):
    function = getattr(np, type(op).__name__.lower())
    return call_numpy_ufunc(function, op, data, **kwargs)


@execute_node.register(ops.Radians, (pd.Series, *numeric_types))
def execute_series_radians(_, data, **kwargs):
    return np.radians(data)


@execute_node.register(ops.Degrees, (pd.Series, *numeric_types))
def execute_series_degrees(_, data, **kwargs):
    return np.degrees(data)


@execute_node.register((ops.Ceil, ops.Floor), pd.Series)
def execute_series_ceil(op, data, **kwargs):
    return_type = np.object_ if data.dtype == np.object_ else np.int64
    func = getattr(np, type(op).__name__.lower())
    return call_numpy_ufunc(func, op, data, **kwargs).astype(return_type)


def vectorize_object(op, arg, *args, **kwargs):
    func = np.vectorize(functools.partial(execute_node, op, **kwargs))
    return pd.Series(func(arg, *args), index=arg.index, name=arg.name)


@execute_node.register(
    ops.Log, pd.Series, (pd.Series, numbers.Real, decimal.Decimal, type(None))
)
def execute_series_log_with_base(op, data, base, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, base, **kwargs)

    if base is None:
        return np.log(data)
    return np.log(data) / np.log(base)


@execute_node.register(ops.Ln, pd.Series)
def execute_series_natural_log(op, data, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return data.apply(functools.partial(execute_node, op, **kwargs))
    return np.log(data)


@execute_node.register(
    ops.Clip,
    pd.Series,
    (pd.Series, type(None)) + numeric_types,
    (pd.Series, type(None)) + numeric_types,
)
def execute_series_clip(op, data, lower, upper, **kwargs):
    return data.clip(lower=lower, upper=upper)


@execute_node.register(ops.Quantile, (pd.Series, SeriesGroupBy), numeric_types)
def execute_series_quantile(op, data, quantile, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data, 'quantile', q=quantile, interpolation=op.interpolation
    )


@execute_node.register(ops.MultiQuantile, pd.Series, np.ndarray)
def execute_series_quantile_multi(
    op, data, quantile, aggcontext=None, **kwargs
):
    result = aggcontext.agg(
        data, 'quantile', q=quantile, interpolation=op.interpolation
    )
    return np.array(result)


@execute_node.register(ops.MultiQuantile, SeriesGroupBy, np.ndarray)
def execute_series_quantile_multi_groupby(
    op, data, quantile, aggcontext=None, **kwargs
):
    def q(x, quantile, interpolation):
        result = x.quantile(quantile, interpolation=interpolation).tolist()
        res = [result for _ in range(len(x))]
        return res

    result = aggcontext.agg(data, q, quantile, op.interpolation)
    return result


@execute_node.register(ops.Cast, type(None), dt.DataType)
def execute_cast_null_to_anything(op, data, type, **kwargs):
    return None


@execute_node.register(ops.Cast, datetime.datetime, dt.String)
def execute_cast_datetime_or_timestamp_to_string(op, data, type, **kwargs):
    """Cast timestamps to strings"""
    return str(data)


@execute_node.register(ops.Cast, datetime.datetime, dt.Int64)
def execute_cast_datetime_to_integer(op, data, type, **kwargs):
    """Cast datetimes to integers"""
    return pd.Timestamp(data).value


@execute_node.register(ops.Cast, pd.Timestamp, dt.Int64)
def execute_cast_timestamp_to_integer(op, data, type, **kwargs):
    """Cast timestamps to integers"""
    return data.value


@execute_node.register(ops.Cast, (np.bool_, bool), dt.Timestamp)
def execute_cast_bool_to_timestamp(op, data, type, **kwargs):
    raise TypeError(
        'Casting boolean values to timestamps does not make sense. If you '
        'really want to cast boolean values to timestamps please cast to '
        'int64 first then to timestamp: '
        "value.cast('int64').cast('timestamp')"
    )


@execute_node.register(ops.Cast, (np.bool_, bool), dt.Interval)
def execute_cast_bool_to_interval(op, data, type, **kwargs):
    raise TypeError(
        'Casting boolean values to intervals does not make sense. If you '
        'really want to cast boolean values to intervals please cast to '
        'int64 first then to interval: '
        "value.cast('int64').cast(ibis.expr.datatypes.Interval(...))"
    )


@execute_node.register(ops.Cast, integer_types + (str,), dt.Timestamp)
def execute_cast_simple_literal_to_timestamp(op, data, type, **kwargs):
    """Cast integer and strings to timestamps"""
    return pd.Timestamp(data, tz=type.timezone)


@execute_node.register(ops.Cast, pd.Timestamp, dt.Timestamp)
def execute_cast_timestamp_to_timestamp(op, data, type, **kwargs):
    """Cast timestamps to other timestamps including timezone if necessary"""
    input_timezone = data.tz
    target_timezone = type.timezone

    if input_timezone == target_timezone:
        return data

    if input_timezone is None or target_timezone is None:
        return data.tz_localize(target_timezone)

    return data.tz_convert(target_timezone)


@execute_node.register(ops.Cast, datetime.datetime, dt.Timestamp)
def execute_cast_datetime_to_datetime(op, data, type, **kwargs):
    return execute_cast_timestamp_to_timestamp(
        op, data, type, **kwargs
    ).to_pydatetime()


@execute_node.register(ops.Cast, fixed_width_types + (str,), dt.DataType)
def execute_cast_string_literal(op, data, type, **kwargs):
    try:
        cast_function = constants.IBIS_TO_PYTHON_LITERAL_TYPES[type]
    except KeyError:
        raise TypeError(f"Don't know how to cast {data!r} to type {type}")
    else:
        return cast_function(data)


@execute_node.register(ops.Round, scalar_types, (int, type(None)))
def execute_round_scalars(op, data, places, **kwargs):
    return round(data, places) if places else round(data)


@execute_node.register(
    ops.Round, pd.Series, (pd.Series, np.integer, type(None), int)
)
def execute_round_series(op, data, places, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, places, **kwargs)
    result = data.round(places or 0)
    return result if places else result.astype('int64')


@execute_node.register(ops.TableColumn, (pd.DataFrame, DataFrameGroupBy))
def execute_table_column_df_or_df_groupby(op, data, **kwargs):
    return data[op.name]


@execute_node.register(
    ops.Aggregation,
    pd.DataFrame,
    tuple,
    tuple,
    tuple,
    tuple,
    tuple,
)
def execute_aggregation_dataframe(
    op,
    data,
    metrics,
    by,
    having,
    predicates,
    sort_keys,
    scope=None,
    timecontext: Optional[TimeContext] = None,
    **kwargs,
):
    assert metrics, 'no metrics found during aggregation execution'

    if sort_keys:
        raise NotImplementedError(
            'sorting on aggregations not yet implemented'
        )

    if predicates:
        predicate = functools.reduce(
            operator.and_,
            (
                execute(p, scope=scope, timecontext=timecontext, **kwargs)
                for p in predicates
            ),
        )
        data = data.loc[predicate]

    columns: Dict[str, str] = {}

    if op.by:
        grouping_key_pairs = list(
            zip(by, map(operator.methodcaller('op'), by))
        )
        grouping_keys = [
            by_op.name
            if isinstance(by_op, ops.TableColumn)
            else execute(
                by, scope=scope, timecontext=timecontext, **kwargs
            ).rename(by.get_name())
            for by, by_op in grouping_key_pairs
        ]
        columns.update(
            (by_op.name, by.get_name())
            for by, by_op in grouping_key_pairs
            if hasattr(by_op, 'name')
        )
        source = data.groupby(grouping_keys)
    else:
        source = data

    scope = scope.merge_scope(Scope({op.table.op(): source}, timecontext))

    pieces = [
        coerce_to_output(
            execute(metric, scope=scope, timecontext=timecontext, **kwargs),
            metric,
        )
        for metric in metrics
    ]

    result = pd.concat(pieces, axis=1)

    # If grouping, need a reset to get the grouping key back as a column
    if by:
        result = result.reset_index()

    result.columns = [columns.get(c, c) for c in result.columns]

    if having:
        # .having(...) is only accessible on groupby, so this should never
        # raise
        if not by:
            raise ValueError(
                'Filtering out aggregation values is not allowed without at '
                'least one grouping key'
            )

        # TODO(phillipc): Don't recompute identical subexpressions
        predicate = functools.reduce(
            operator.and_,
            (
                execute(h, scope=scope, timecontext=timecontext, **kwargs)
                for h in having
            ),
        )
        assert len(predicate) == len(
            result
        ), 'length of predicate does not match length of DataFrame'
        result = result.loc[predicate.values]
    return result


@execute_node.register(ops.Reduction, SeriesGroupBy, type(None))
def execute_reduction_series_groupby(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, type(op).__name__.lower())


variance_ddof = {'pop': 0, 'sample': 1}


@execute_node.register(ops.Variance, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_var(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'var', ddof=variance_ddof[op.how])


@execute_node.register(ops.StandardDev, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_std(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'std', ddof=variance_ddof[op.how])


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    SeriesGroupBy,
    type(None),
)
def execute_count_distinct_series_groupby(
    op, data, _, aggcontext=None, **kwargs
):
    return aggcontext.agg(data, 'nunique')


@execute_node.register(ops.Arbitrary, SeriesGroupBy, type(None))
def execute_arbitrary_series_groupby(op, data, _, aggcontext=None, **kwargs):
    how = op.how
    if how is None:
        how = 'first'

    if how not in {'first', 'last'}:
        raise com.OperationNotDefinedError(
            f'Arbitrary {how!r} is not supported'
        )
    return aggcontext.agg(data, how)


@execute_node.register(
    (ops.ArgMin, ops.ArgMax),
    SeriesGroupBy,
    SeriesGroupBy,
    type(None),
)
def execute_reduction_series_groupby_argidx(
    op, data, key, _, aggcontext=None, **kwargs
):
    method = operator.methodcaller(op.__class__.__name__.lower())

    def reduce(data, key=key.obj, method=method):
        return data.iloc[method(key.loc[data.index])]

    return aggcontext.agg(data, reduce)


def _filtered_reduction(mask, method, data):
    return method(data[mask[data.index]])


@execute_node.register(ops.Reduction, SeriesGroupBy, SeriesGroupBy)
def execute_reduction_series_gb_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    method = operator.methodcaller(type(op).__name__.lower())
    return aggcontext.agg(
        data, functools.partial(_filtered_reduction, mask.obj, method)
    )


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    SeriesGroupBy,
    SeriesGroupBy,
)
def execute_count_distinct_series_groupby_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(
        data,
        functools.partial(_filtered_reduction, mask.obj, pd.Series.nunique),
    )


@execute_node.register(ops.Variance, SeriesGroupBy, SeriesGroupBy)
def execute_var_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data,
        lambda x, mask=mask.obj, ddof=variance_ddof[op.how]: (
            x[mask[x.index]].var(ddof=ddof)
        ),
    )


@execute_node.register(ops.StandardDev, SeriesGroupBy, SeriesGroupBy)
def execute_std_series_groupby_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data,
        lambda x, mask=mask.obj, ddof=variance_ddof[op.how]: (
            x[mask[x.index]].std(ddof=ddof)
        ),
    )


@execute_node.register(ops.Count, DataFrameGroupBy, type(None))
def execute_count_frame_groupby(op, data, _, **kwargs):
    result = data.size()
    # FIXME(phillipc): We should not hard code this column name
    result.name = 'count'
    return result


@execute_node.register(ops.Reduction, pd.Series, (pd.Series, type(None)))
def execute_reduction_series_mask(op, data, mask, aggcontext=None, **kwargs):
    operand = data[mask] if mask is not None else data
    return aggcontext.agg(operand, type(op).__name__.lower())


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    pd.Series,
    (pd.Series, type(None)),
)
def execute_count_distinct_series_mask(
    op, data, mask, aggcontext=None, **kwargs
):
    return aggcontext.agg(data[mask] if mask is not None else data, 'nunique')


@execute_node.register(ops.Arbitrary, pd.Series, (pd.Series, type(None)))
def execute_arbitrary_series_mask(op, data, mask, aggcontext=None, **kwargs):
    if op.how == 'first':
        index = 0
    elif op.how == 'last':
        index = -1
    else:
        raise com.OperationNotDefinedError(
            f'Arbitrary {op.how!r} is not supported'
        )

    data = data[mask] if mask is not None else data
    return data.iloc[index]


@execute_node.register(ops.StandardDev, pd.Series, (pd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        'std',
        ddof=variance_ddof[op.how],
    )


@execute_node.register(ops.Variance, pd.Series, (pd.Series, type(None)))
def execute_variance_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        'var',
        ddof=variance_ddof[op.how],
    )


@execute_node.register((ops.Any, ops.All), (pd.Series, SeriesGroupBy))
def execute_any_all_series(op, data, aggcontext=None, **kwargs):
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = aggcontext.agg(data, type(op).__name__.lower())
    else:
        result = aggcontext.agg(
            data, lambda data: getattr(data, type(op).__name__.lower())()
        )
    try:
        return result.astype(bool)
    except TypeError:
        return result


@execute_node.register(ops.NotAny, (pd.Series, SeriesGroupBy))
def execute_notany_series(op, data, aggcontext=None, **kwargs):
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = ~(aggcontext.agg(data, 'any'))
    else:
        result = aggcontext.agg(data, lambda data: ~(data.any()))
    try:
        return result.astype(bool)
    except TypeError:
        return result


@execute_node.register(ops.NotAll, (pd.Series, SeriesGroupBy))
def execute_notall_series(op, data, aggcontext=None, **kwargs):
    if isinstance(aggcontext, (agg_ctx.Summarize, agg_ctx.Transform)):
        result = ~(aggcontext.agg(data, 'all'))
    else:
        result = aggcontext.agg(data, lambda data: ~(data.all()))
    try:
        return result.astype(bool)
    except TypeError:
        return result


@execute_node.register(ops.Count, pd.DataFrame, type(None))
def execute_count_frame(op, data, _, **kwargs):
    return len(data)


@execute_node.register(ops.BitAnd, pd.Series, (pd.Series, type(None)))
def execute_bit_and_series(_, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        np.bitwise_and.reduce,
    )


@execute_node.register(ops.BitOr, pd.Series, (pd.Series, type(None)))
def execute_bit_or_series(_, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        np.bitwise_or.reduce,
    )


@execute_node.register(ops.BitXor, pd.Series, (pd.Series, type(None)))
def execute_bit_xor_series(_, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        np.bitwise_xor.reduce,
    )


@execute_node.register(
    (ops.ArgMin, ops.ArgMax),
    pd.Series,
    pd.Series,
    (pd.Series, type(None)),
)
def execute_argmin_series_mask(op, data, key, mask, aggcontext=None, **kwargs):
    method_name = op.__class__.__name__.lower()
    masked_key = key[mask] if mask is not None else key
    idx = aggcontext.agg(masked_key, method_name)
    masked = data[mask] if mask is not None else data
    return masked.iloc[idx]


@execute_node.register((ops.Not, ops.Negate), (bool, np.bool_))
def execute_not_bool(_, data, **kwargs):
    return not data


def _execute_binary_op_impl(op, left, right, **_):
    op_type = type(op)
    try:
        operation = constants.BINARY_OPERATIONS[op_type]
    except KeyError:
        raise NotImplementedError(
            f'Binary operation {op_type.__name__} not implemented'
        )
    else:
        return operation(left, right)


@execute_node.register(ops.Binary, pd.Series, pd.Series)
@execute_node.register(
    (ops.NumericBinary, ops.LogicalBinary, ops.Comparison),
    numeric_types,
    pd.Series,
)
@execute_node.register(
    (ops.NumericBinary, ops.LogicalBinary, ops.Comparison),
    pd.Series,
    numeric_types,
)
@execute_node.register(
    (ops.NumericBinary, ops.LogicalBinary, ops.Comparison),
    numeric_types,
    numeric_types,
)
@execute_node.register((ops.Comparison, ops.Add, ops.Multiply), pd.Series, str)
@execute_node.register((ops.Comparison, ops.Add, ops.Multiply), str, pd.Series)
@execute_node.register((ops.Comparison, ops.Add), str, str)
@execute_node.register(ops.Multiply, integer_types, str)
@execute_node.register(ops.Multiply, str, integer_types)
@execute_node.register(ops.Comparison, pd.Series, timestamp_types)
@execute_node.register(ops.Comparison, timedelta_types, pd.Series)
def execute_binary_op(op, left, right, **kwargs):
    return _execute_binary_op_impl(op, left, right, **kwargs)


@execute_node.register(ops.Comparison, pd.Series, date_types)
def execute_binary_op_date(op, left, right, **kwargs):
    return _execute_binary_op_impl(
        op, pd.to_datetime(left), pd.to_datetime(right), **kwargs
    )


@execute_node.register(ops.Binary, SeriesGroupBy, SeriesGroupBy)
def execute_binary_op_series_group_by(op, left, right, **kwargs):
    left_groupings = left.grouper.groupings
    right_groupings = right.grouper.groupings
    if left_groupings != right_groupings:
        raise ValueError(
            'Cannot perform {} operation on two series with '
            'different groupings'.format(type(op).__name__)
        )
    result = execute_binary_op(op, left.obj, right.obj, **kwargs)
    return result.groupby(left_groupings)


@execute_node.register(ops.Binary, SeriesGroupBy, simple_types)
def execute_binary_op_series_gb_simple(op, left, right, **kwargs):
    result = execute_binary_op(op, left.obj, right, **kwargs)
    return result.groupby(left.grouper.groupings)


@execute_node.register(ops.Binary, simple_types, SeriesGroupBy)
def execute_binary_op_simple_series_gb(op, left, right, **kwargs):
    result = execute_binary_op(op, left, right.obj, **kwargs)
    return result.groupby(right.grouper.groupings)


@execute_node.register(ops.Unary, SeriesGroupBy)
def execute_unary_op_series_gb(op, operand, **kwargs):
    result = execute_node(op, operand.obj, **kwargs)
    return result.groupby(operand.grouper.groupings)


@execute_node.register(
    (ops.Log, ops.Round),
    SeriesGroupBy,
    (numbers.Real, decimal.Decimal, type(None)),
)
def execute_log_series_gb_others(op, left, right, **kwargs):
    result = execute_node(op, left.obj, right, **kwargs)
    return result.groupby(left.grouper.groupings)


@execute_node.register((ops.Log, ops.Round), SeriesGroupBy, SeriesGroupBy)
def execute_log_series_gb_series_gb(op, left, right, **kwargs):
    result = execute_node(op, left.obj, right.obj, **kwargs)
    return result.groupby(left.grouper.groupings)


@execute_node.register(ops.Not, pd.Series)
def execute_not_series(op, data, **kwargs):
    return ~data


@execute_node.register(ops.NullIfZero, pd.Series)
def execute_null_if_zero_series(op, data, **kwargs):
    return data.where(data != 0, np.nan)


@execute_node.register(ops.StringSplit, pd.Series, (pd.Series, str))
def execute_string_split(op, data, delimiter, **kwargs):
    # Doing the iteration using `map` is much faster than doing the iteration
    # using `Series.apply` due to Pandas-related overhead.
    return pd.Series(map(lambda s: np.array(s.split(delimiter)), data))


@execute_node.register(
    ops.Between,
    pd.Series,
    (pd.Series, numbers.Real, str, datetime.datetime),
    (pd.Series, numbers.Real, str, datetime.datetime),
)
def execute_between(op, data, lower, upper, **kwargs):
    return data.between(lower, upper)


@execute_node.register(ops.Union, pd.DataFrame, pd.DataFrame, bool)
def execute_union_dataframe_dataframe(
    op, left: pd.DataFrame, right: pd.DataFrame, distinct, **kwargs
):
    result = pd.concat([left, right], axis=0)
    return result.drop_duplicates() if distinct else result


@execute_node.register(ops.Intersection, pd.DataFrame, pd.DataFrame, bool)
def execute_intersection_dataframe_dataframe(
    op,
    left: pd.DataFrame,
    right: pd.DataFrame,
    distinct: bool,
    **kwargs,
):
    if not distinct:
        raise NotImplementedError(
            "`distinct=False` is not supported by the pandas backend"
        )
    result = left.merge(right, on=list(left.columns), how="inner")
    return result


@execute_node.register(ops.Difference, pd.DataFrame, pd.DataFrame, bool)
def execute_difference_dataframe_dataframe(
    op,
    left: pd.DataFrame,
    right: pd.DataFrame,
    distinct: bool,
    **kwargs,
):
    if not distinct:
        raise NotImplementedError(
            "`distinct=False` is not supported by the pandas backend"
        )
    merged = left.merge(
        right, on=list(left.columns), how="outer", indicator=True
    )
    result = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)
    return result


@execute_node.register(ops.IsNull, pd.Series)
def execute_series_isnull(op, data, **kwargs):
    return data.isnull()


@execute_node.register(ops.NotNull, pd.Series)
def execute_series_notnnull(op, data, **kwargs):
    return data.notnull()


@execute_node.register(ops.IsNan, (pd.Series, floating_types))
def execute_isnan(op, data, **kwargs):
    try:
        return np.isnan(data)
    except (TypeError, ValueError):
        # if `data` contains `None` np.isnan will complain
        # so we take advantage of NaN not equaling itself
        # to do the correct thing
        return data != data


@execute_node.register(ops.IsInf, (pd.Series, floating_types))
def execute_isinf(op, data, **kwargs):
    return np.isinf(data)


@execute_node.register(ops.SelfReference, pd.DataFrame)
def execute_node_self_reference_dataframe(op, data, **kwargs):
    return data


@execute_node.register(ops.Alias, object)
def execute_alias(op, data, **kwargs):
    # just return the underlying argument because the naming is handled
    # by the translator for the top level expression
    return data


@execute_node.register(ops.ValueList, collections.abc.Sequence)
def execute_node_value_list(op, _, **kwargs):
    return [execute(arg, **kwargs) for arg in op.values]


@execute_node.register(ops.StringConcat, collections.abc.Sequence)
def execute_node_string_concat(op, args, **kwargs):
    return functools.reduce(operator.add, args)


@execute_node.register(ops.StringJoin, collections.abc.Sequence)
def execute_node_string_join(op, args, **kwargs):
    return op.sep.join(args)


@execute_node.register(
    ops.Contains,
    pd.Series,
    (collections.abc.Sequence, collections.abc.Set, pd.Series),
)
def execute_node_contains_series_sequence(op, data, elements, **kwargs):
    return data.isin(elements)


@execute_node.register(
    ops.Contains,
    SeriesGroupBy,
    (collections.abc.Sequence, collections.abc.Set, pd.Series),
)
def execute_node_contains_series_group_by_sequence(
    op, data, elements, **kwargs
):
    return data.obj.isin(elements).groupby(data.grouper.groupings)


@execute_node.register(
    ops.NotContains,
    pd.Series,
    (collections.abc.Sequence, collections.abc.Set, pd.Series),
)
def execute_node_not_contains_series_sequence(op, data, elements, **kwargs):
    return ~(data.isin(elements))


@execute_node.register(
    ops.NotContains,
    SeriesGroupBy,
    (collections.abc.Sequence, collections.abc.Set, pd.Series),
)
def execute_node_not_contains_series_group_by_sequence(
    op, data, elements, **kwargs
):
    return (~data.obj.isin(elements)).groupby(data.grouper.groupings)


def pd_where(cond, true, false):
    """Execute `where` following ibis's intended semantics"""
    if isinstance(cond, pd.Series):
        if not isinstance(true, pd.Series):
            true = pd.Series(
                np.repeat(true, len(cond)), name=cond.name, index=cond.index
            )
        return true.where(cond, other=false)
    if cond:
        if isinstance(false, pd.Series) and not isinstance(true, pd.Series):
            return pd.Series(np.repeat(true, len(false)))
        return true
    else:
        if isinstance(true, pd.Series) and not isinstance(false, pd.Series):
            return pd.Series(np.repeat(false, len(true)), index=true.index)
        return false


@execute_node.register(
    ops.Where, (pd.Series, *boolean_types), pd.Series, pd.Series
)
@execute_node.register(
    ops.Where, (pd.Series, *boolean_types), pd.Series, simple_types
)
@execute_node.register(
    ops.Where, (pd.Series, *boolean_types), simple_types, pd.Series
)
@execute_node.register(
    ops.Where, (pd.Series, *boolean_types), type(None), type(None)
)
def execute_node_where(op, cond, true, false, **kwargs):
    return pd_where(cond, true, false)


# For true/false as scalars, we only support identical type pairs + None to
# limit the size of the dispatch table and not have to worry about type
# promotion.
for typ in (str, *scalar_types):
    for cond_typ in (pd.Series, *boolean_types):
        execute_node.register(ops.Where, cond_typ, typ, typ)(
            execute_node_where
        )
        execute_node.register(ops.Where, cond_typ, type(None), typ)(
            execute_node_where
        )
        execute_node.register(ops.Where, cond_typ, typ, type(None))(
            execute_node_where
        )


@execute_node.register(PandasTable, PandasBackend)
def execute_database_table_client(
    op, client, timecontext: Optional[TimeContext], **kwargs
):
    df = client.dictionary[op.name]
    if timecontext:
        begin, end = timecontext
        time_col = get_time_col()
        if time_col not in df:
            raise com.IbisError(
                f'Table {op.name} must have a time column named {time_col}'
                ' to execute with time context.'
            )
        # filter with time context
        mask = df[time_col].between(begin, end)
        return df.loc[mask].reset_index(drop=True)
    return df


MATH_FUNCTIONS = {
    ops.Floor: math.floor,
    ops.Ln: math.log,
    ops.Log2: lambda x: math.log(x, 2),
    ops.Log10: math.log10,
    ops.Exp: math.exp,
    ops.Sqrt: math.sqrt,
    ops.Abs: abs,
    ops.Ceil: math.ceil,
    ops.Sign: lambda x: 0 if not x else -1 if x < 0 else 1,
}

MATH_FUNCTION_TYPES = tuple(MATH_FUNCTIONS.keys())


@execute_node.register(MATH_FUNCTION_TYPES, numeric_types)
def execute_node_math_function_number(op, value, **kwargs):
    return MATH_FUNCTIONS[type(op)](value)


@execute_node.register(ops.Log, numeric_types, numeric_types)
def execute_node_log_number_number(op, value, base, **kwargs):
    return math.log(value, base)


@execute_node.register(ops.DropNa, pd.DataFrame, type(None))
@execute_node.register(ops.DropNa, pd.DataFrame, tuple)
def execute_node_dropna_dataframe(op, df, subset, **kwargs):
    if subset is not None:
        subset = [col.get_name() for col in subset]
    return df.dropna(how=op.how, subset=subset)


@execute_node.register(ops.FillNa, pd.DataFrame, simple_types)
def execute_node_fillna_dataframe_scalar(op, df, replacements, **kwargs):
    return df.fillna(replacements)


@execute_node.register(ops.FillNa, pd.DataFrame)
def execute_node_fillna_dataframe_dict(op, df, **kwargs):
    return df.fillna(dict(op.replacements))


@execute_node.register(ops.IfNull, pd.Series, simple_types)
@execute_node.register(ops.IfNull, pd.Series, pd.Series)
def execute_node_ifnull_series(op, value, replacement, **kwargs):
    return value.fillna(replacement)


@execute_node.register(ops.IfNull, simple_types, pd.Series)
def execute_node_ifnull_scalar_series(op, value, replacement, **kwargs):
    return (
        replacement
        if pd.isnull(value)
        else pd.Series(value, index=replacement.index)
    )


@execute_node.register(ops.IfNull, simple_types, simple_types)
def execute_node_if_scalars(op, value, replacement, **kwargs):
    return replacement if pd.isnull(value) else value


@execute_node.register(ops.NullIf, simple_types, simple_types)
def execute_node_nullif_scalars(op, value1, value2, **kwargs):
    return np.nan if value1 == value2 else value1


@execute_node.register(ops.NullIf, pd.Series, pd.Series)
def execute_node_nullif_series(op, series1, series2, **kwargs):
    return series1.where(series1 != series2)


@execute_node.register(ops.NullIf, pd.Series, simple_types)
def execute_node_nullif_series_scalar(op, series, value, **kwargs):
    return series.where(series != value)


@execute_node.register(ops.NullIf, simple_types, pd.Series)
def execute_node_nullif_scalar_series(op, value, series, **kwargs):
    return pd.Series(
        np.where(series.values == value, np.nan, value), index=series.index
    )


def coalesce(values):
    return functools.reduce(
        lambda a1, a2: np.where(pd.isnull(a1), a2, a1),
        values,
    )


@toolz.curry
def promote_to_sequence(length, obj):
    return obj.values if isinstance(obj, pd.Series) else np.repeat(obj, length)


def compute_row_reduction(func, value, **kwargs):
    final_sizes = {len(x) for x in value if isinstance(x, Sized)}
    if not final_sizes:
        return func(value)
    (final_size,) = final_sizes
    raw = func(list(map(promote_to_sequence(final_size), value)), **kwargs)
    return pd.Series(raw).squeeze()


@execute_node.register(ops.Greatest, collections.abc.Sequence)
def execute_node_greatest_list(op, value, **kwargs):
    return compute_row_reduction(np.maximum.reduce, value, axis=0)


@execute_node.register(ops.Least, collections.abc.Sequence)
def execute_node_least_list(op, value, **kwargs):
    return compute_row_reduction(np.minimum.reduce, value, axis=0)


@execute_node.register(ops.Coalesce, collections.abc.Sequence)
def execute_node_coalesce(op, values, **kwargs):
    # TODO: this is slow
    return compute_row_reduction(coalesce, values)


def wrap_case_result(raw, expr):
    """Wrap a CASE statement result in a Series and handle returning scalars.

    Parameters
    ----------
    raw : ndarray[T]
        The raw results of executing the ``CASE`` expression
    expr : Value
        The expression from the which `raw` was computed

    Returns
    -------
    Union[scalar, Series]
    """
    raw_1d = np.atleast_1d(raw)
    if np.any(pd.isnull(raw_1d)):
        result = pd.Series(raw_1d)
    else:
        result = pd.Series(
            raw_1d, dtype=constants.IBIS_TYPE_TO_PANDAS_TYPE[expr.type()]
        )
    if result.size == 1 and isinstance(expr, ir.Scalar):
        return result.iloc[0].item()
    return result


@execute_node.register(ops.SearchedCase, list, list, object)
def execute_searched_case(op, whens, thens, otherwise, **kwargs):
    if otherwise is None:
        otherwise = np.nan
    raw = np.select(whens, thens, otherwise)
    return wrap_case_result(raw, op.to_expr())


@execute_node.register(ops.SimpleCase, object, list, list, object)
def execute_simple_case_scalar(op, value, whens, thens, otherwise, **kwargs):
    if otherwise is None:
        otherwise = np.nan
    raw = np.select(np.asarray(whens) == value, thens, otherwise)
    return wrap_case_result(raw, op.to_expr())


@execute_node.register(ops.SimpleCase, pd.Series, list, list, object)
def execute_simple_case_series(op, value, whens, thens, otherwise, **kwargs):
    if otherwise is None:
        otherwise = np.nan
    raw = np.select([value == when for when in whens], thens, otherwise)
    return wrap_case_result(raw, op.to_expr())


@execute_node.register(ops.Distinct, pd.DataFrame)
def execute_distinct_dataframe(op, df, **kwargs):
    return df.drop_duplicates()


@execute_node.register(ops.RowID)
def execute_rowid(op, *args, **kwargs):
    raise com.UnsupportedOperationError(
        'rowid is not supported in pandas backends'
    )


@execute_node.register(ops.TableArrayView, pd.DataFrame)
def execute_table_array_view(op, _, **kwargs):
    return execute(op.table).squeeze()


@execute_node.register(ops.ZeroIfNull, pd.Series)
def execute_zero_if_null_series(op, data, **kwargs):
    zero = op.arg.type().to_pandas().type(0)
    return data.replace({np.nan: zero, None: zero, pd.NA: zero})


@execute_node.register(
    ops.ZeroIfNull,
    (type(None), type(pd.NA), numbers.Real, np.integer, np.floating),
)
def execute_zero_if_null_scalar(op, data, **kwargs):
    if data is None or pd.isna(data) or math.isnan(data) or np.isnan(data):
        return op.arg.type().to_pandas().type(0)
    return data
