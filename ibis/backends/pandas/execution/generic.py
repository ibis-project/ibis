"""Execution rules for generic ibis operations."""

from __future__ import annotations

import collections
import contextlib
import datetime
import decimal
import functools
import math
import numbers
import operator
from collections.abc import Mapping, Sized

import numpy as np
import pandas as pd
import pytz
import toolz
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.df.scope import Scope
from ibis.backends.base.df.timecontext import TimeContext, get_time_col
from ibis.backends.pandas import Backend as PandasBackend
from ibis.backends.pandas import aggcontext as agg_ctx
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
from ibis.backends.pandas.execution.util import coerce_to_output, get_grouping


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
    if value is None:
        return value
    return int(value)


@execute_literal.register(ops.Literal, object, dt.Boolean)
def execute_node_literal_any_boolean_datatype(op, value, datatype, **kwargs):
    if value is None:
        return value
    return bool(value)


@execute_literal.register(ops.Literal, object, dt.Floating)
def execute_node_literal_any_floating_datatype(op, value, datatype, **kwargs):
    if value is None:
        return value
    return float(value)


@execute_literal.register(ops.Literal, object, dt.Array)
def execute_node_literal_any_array_datatype(op, value, datatype, **kwargs):
    if value is None:
        return value
    return np.array(value)


@execute_literal.register(ops.Literal, dt.DataType)
def execute_node_literal_datatype(op, datatype, **kwargs):
    return op.value


@execute_literal.register(
    ops.Literal, (*timedelta_types, str, *integer_types, type(None)), dt.Interval
)
def execute_interval_literal(op, value, dtype, **kwargs):
    if value is None:
        return pd.NaT
    return pd.Timedelta(value, dtype.unit.short)


@execute_node.register(ops.Limit, pd.DataFrame, integer_types, integer_types)
def execute_limit_frame(op, data, nrows: int, offset: int, **kwargs):
    return data.iloc[offset : offset + nrows]


@execute_node.register(ops.Limit, pd.DataFrame, type(None), integer_types)
def execute_limit_frame_no_limit(op, data, nrows: None, offset: int, **kwargs):
    return data.iloc[offset:]


@execute_node.register(ops.Cast, SeriesGroupBy, dt.DataType)
def execute_cast_series_group_by(op, data, type, **kwargs):
    result = execute_cast_series_generic(op, data.obj, type, **kwargs)
    return result.groupby(get_grouping(data.grouper.groupings), group_keys=False)


@execute_node.register(ops.Cast, pd.Series, dt.DataType)
def execute_cast_series_generic(op, data, type, **kwargs):
    out = data.astype(constants.IBIS_TYPE_TO_PANDAS_TYPE[type])
    if type.is_integer():
        if op.arg.dtype.is_timestamp():
            return out.floordiv(int(1e9))
        elif op.arg.dtype.is_date():
            return out.floordiv(int(24 * 60 * 60 * 1e9))
    return out


@execute_node.register(ops.Cast, pd.Series, dt.Array)
def execute_cast_series_array(op, data, type, **kwargs):
    value_type = type.value_type
    numpy_type = constants.IBIS_TYPE_TO_PANDAS_TYPE.get(value_type, None)
    if numpy_type is None:
        raise ValueError(
            "Array value type must be a primitive type "
            "(e.g., number, string, or timestamp)"
        )

    def cast_to_array(array, numpy_type=numpy_type):
        elems = [
            el if el is None else np.array(el, dtype=numpy_type).item() for el in array
        ]
        try:
            return np.array(elems, dtype=numpy_type)
        except TypeError:
            return np.array(elems)

    return data.map(cast_to_array)


@execute_node.register(ops.Cast, pd.Series, dt.Timestamp)
def execute_cast_series_timestamp(op, data, type, **kwargs):
    arg = op.arg
    from_type = arg.dtype

    if from_type.equals(type):  # noop cast
        return data

    tz = type.timezone

    if from_type.is_timestamp():
        from_tz = from_type.timezone
        if tz is None and from_tz is None:
            return data
        elif tz is None or from_tz is None:
            return data.dt.tz_localize(tz)
        elif tz is not None and from_tz is not None:
            return data.dt.tz_convert(tz)
    elif from_type.is_date():
        return data if tz is None else data.dt.tz_localize(tz)

    if from_type.is_string() or from_type.is_integer():
        if from_type.is_integer():
            timestamps = pd.to_datetime(data.values, unit="s")
        else:
            timestamps = pd.to_datetime(data.values)
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
    from_type = arg.dtype

    if from_type.equals(type):
        return data

    if from_type.is_timestamp():
        return _normalize(
            data.values, data.index, data.name, timezone=from_type.timezone
        )

    if from_type.is_string():
        values = data.values
        datetimes = pd.to_datetime(values)
        with contextlib.suppress(TypeError):
            datetimes = datetimes.tz_convert(None)
        dates = _normalize(datetimes, data.index, data.name)
        return pd.Series(dates, index=data.index, name=data.name)

    if from_type.is_integer():
        return pd.Series(
            pd.to_datetime(data.values, unit="D").values,
            index=data.index,
            name=data.name,
        )

    raise TypeError(f"Don't know how to cast {from_type} to {type}")


@execute_node.register(ops.SortKey, pd.Series, bool)
def execute_sort_key_series(op, data, _, **kwargs):
    return data


def call_numpy_ufunc(func, op, data, **kwargs):
    if getattr(data, "dtype", None) == np.dtype(np.object_):
        return data.apply(functools.partial(execute_node, op, **kwargs))
    if func is None:
        raise com.OperationNotDefinedError(f"{type(op).__name__} not supported")
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
        get_grouping(data.grouper.groupings), group_keys=False
    )


@execute_node.register(ops.Unary, pd.Series)
def execute_series_unary_op(op, data, **kwargs):
    op_type = type(op)
    if op_type == ops.BitwiseNot:
        function = np.bitwise_not
    else:
        function = getattr(np, op_type.__name__.lower())
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
    return 1.0 / np.tan(data)


@execute_node.register(
    ops.Atan2, (pd.Series, *numeric_types), (pd.Series, *numeric_types)
)
def execute_series_atan2(_, y, x, **kwargs):
    return np.arctan2(y, x)


@execute_node.register((ops.Cos, ops.Sin, ops.Tan), (pd.Series, *numeric_types))
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


@execute_node.register(ops.BitwiseNot, integer_types)
def execute_int_bitwise_not(op, data, **kwargs):
    return np.invert(data)


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


@execute_node.register(
    ops.Quantile,
    pd.Series,
    (np.ndarray, *numeric_types),
    (pd.Series, type(None)),
)
def execute_series_quantile(op, data, quantile, mask, aggcontext=None, **_):
    return aggcontext.agg(
        data if mask is None else data.loc[mask],
        "quantile",
        q=quantile,
    )


@execute_node.register(ops.Quantile, pd.Series, (np.ndarray, *numeric_types))
def execute_series_quantile_default(op, data, quantile, aggcontext=None, **_):
    return aggcontext.agg(data, "quantile", q=quantile)


@execute_node.register(
    ops.Quantile,
    SeriesGroupBy,
    (np.ndarray, *numeric_types),
    (SeriesGroupBy, type(None)),
)
def execute_series_group_by_quantile(op, data, quantile, mask, aggcontext=None, **_):
    return aggcontext.agg(
        data,
        (
            "quantile"
            if mask is None
            else functools.partial(_filtered_reduction, mask.obj, pd.Series.quantile)
        ),
        q=quantile,
    )


@execute_node.register(
    ops.MultiQuantile,
    pd.Series,
    (np.ndarray, *numeric_types),
    (pd.Series, type(None)),
)
def execute_series_quantile_multi(op, data, quantile, mask, aggcontext=None, **_):
    return np.array(
        aggcontext.agg(data if mask is None else data.loc[mask], "quantile", q=quantile)
    )


@execute_node.register(
    ops.MultiQuantile,
    SeriesGroupBy,
    np.ndarray,
    (SeriesGroupBy, type(None)),
)
def execute_series_quantile_multi_groupby(
    op, data, quantile, mask, aggcontext=None, **kwargs
):
    def q(x, quantile):
        result = x.quantile(quantile).tolist()
        return [result for _ in range(len(x))]

    return aggcontext.agg(
        data,
        q if mask is None else functools.partial(_filtered_reduction, mask.obj, q),
        quantile,
    )


@execute_node.register(ops.MultiQuantile, SeriesGroupBy, np.ndarray)
def execute_series_quantile_multi_groupby_default(
    op, data, quantile, aggcontext=None, **_
):
    def q(x, quantile):
        result = x.quantile(quantile).tolist()
        return [result for _ in range(len(x))]

    return aggcontext.agg(data, q, quantile)


@execute_node.register(ops.Cast, type(None), dt.DataType)
def execute_cast_null_to_anything(op, data, type, **kwargs):
    return None


@execute_node.register(ops.Cast, datetime.datetime, dt.String)
def execute_cast_datetime_or_timestamp_to_string(op, data, type, **kwargs):
    """Cast timestamps to strings."""
    return str(data)


@execute_node.register(ops.Cast, datetime.datetime, dt.Int64)
def execute_cast_timestamp_to_integer(op, data, type, **kwargs):
    """Cast timestamps to integers."""
    t = pd.Timestamp(data)
    return pd.NA if pd.isna(t) else int(t.timestamp())


@execute_node.register(ops.Cast, (np.bool_, bool), dt.Timestamp)
def execute_cast_bool_to_timestamp(op, data, type, **kwargs):
    raise TypeError(
        "Casting boolean values to timestamps does not make sense. If you "
        "really want to cast boolean values to timestamps please cast to "
        "int64 first then to timestamp: "
        "value.cast('int64').cast('timestamp')"
    )


@execute_node.register(ops.Cast, (np.bool_, bool), dt.Interval)
def execute_cast_bool_to_interval(op, data, type, **kwargs):
    raise TypeError(
        "Casting boolean values to intervals does not make sense. If you "
        "really want to cast boolean values to intervals please cast to "
        "int64 first then to interval: "
        "value.cast('int64').cast(ibis.expr.datatypes.Interval(...))"
    )


@execute_node.register(ops.Cast, integer_types, dt.Timestamp)
def execute_cast_integer_to_timestamp(op, data, type, **kwargs):
    """Cast integer to timestamp."""
    return pd.Timestamp(data, unit="s", tz=type.timezone)


@execute_node.register(ops.Cast, str, dt.Timestamp)
def execute_cast_string_to_timestamp(op, data, type, **kwargs):
    """Cast string to timestamp."""
    return pd.Timestamp(data, tz=type.timezone)


@execute_node.register(ops.Cast, datetime.datetime, dt.Timestamp)
def execute_cast_timestamp_to_timestamp(op, data, type, **kwargs):
    """Cast timestamps to other timestamps including timezone if necessary."""
    input_timezone = data.tzinfo
    target_timezone = type.timezone

    if input_timezone == target_timezone:
        return data

    if input_timezone is None or target_timezone is None:
        return data.astimezone(
            tz=None if target_timezone is None else pytz.timezone(target_timezone)
        )

    return data.astimezone(tz=pytz.timezone(target_timezone))


@execute_node.register(ops.Cast, fixed_width_types + (str,), dt.DataType)
def execute_cast_string_literal(op, data, type, **kwargs):
    try:
        cast_function = constants.IBIS_TO_PYTHON_LITERAL_TYPES[type]
    except KeyError:
        raise TypeError(f"Don't know how to cast {data!r} to type {type}")
    else:
        return cast_function(data)


@execute_node.register(ops.Cast, Mapping, dt.DataType)
def execute_cast_mapping_literal(op, data, type, **kwargs):
    data = (
        (ops.Literal(k, type.key_type), ops.Literal(v, type.value_type))
        for k, v in data.items()
    )
    return {execute(k, **kwargs): execute(v, **kwargs) for k, v in data}


@execute_node.register(ops.Round, scalar_types, (int, type(None)))
def execute_round_scalars(op, data, places, **kwargs):
    return round(data, places) if places else round(data)


@execute_node.register(ops.Round, pd.Series, (pd.Series, np.integer, type(None), int))
def execute_round_series(op, data, places, **kwargs):
    if data.dtype == np.dtype(np.object_):
        return vectorize_object(op, data, places, **kwargs)
    result = data.round(places or 0)
    return result if places else result.astype("int64")


@execute_node.register(ops.TableColumn, (pd.DataFrame, DataFrameGroupBy))
def execute_table_column_df_or_df_groupby(op, data, **kwargs):
    return data[op.name]


@execute_node.register(ops.Aggregation, pd.DataFrame)
def execute_aggregation_dataframe(
    op,
    data,
    scope=None,
    timecontext: TimeContext | None = None,
    **kwargs,
):
    assert op.metrics, "no metrics found during aggregation execution"

    if op.sort_keys:
        raise NotImplementedError("sorting on aggregations not yet implemented")

    if op.predicates:
        predicate = functools.reduce(
            operator.and_,
            (
                execute(p, scope=scope, timecontext=timecontext, **kwargs)
                for p in op.predicates
            ),
        )
        data = data.loc[predicate]

    columns: dict[str, str] = {}

    if op.by:
        grouping_keys = [
            key.name
            if isinstance(key, ops.TableColumn)
            else execute(key, scope=scope, timecontext=timecontext, **kwargs).rename(
                key.name
            )
            for key in op.by
        ]
        source = data.groupby(
            grouping_keys[0] if len(grouping_keys) == 1 else grouping_keys,
            group_keys=False,
        )
    else:
        source = data

    scope = scope.merge_scope(Scope({op.table: source}, timecontext))

    pieces = [
        coerce_to_output(
            execute(metric, scope=scope, timecontext=timecontext, **kwargs),
            metric,
        )
        for metric in op.metrics
    ]

    result = pd.concat(pieces, axis=1)

    # If grouping, need a reset to get the grouping key back as a column
    if op.by:
        result = result.reset_index()

    result.columns = [columns.get(c, c) for c in result.columns]

    if op.having:
        # .having(...) is only accessible on groupby, so this should never
        # raise
        if not op.by:
            raise ValueError(
                "Filtering out aggregation values is not allowed without at "
                "least one grouping key"
            )

        # TODO(phillipc): Don't recompute identical subexpressions
        predicate = functools.reduce(
            operator.and_,
            (
                execute(h, scope=scope, timecontext=timecontext, **kwargs)
                for h in op.having
            ),
        )
        assert len(predicate) == len(
            result
        ), "length of predicate does not match length of DataFrame"
        result = result.loc[predicate.values]
    return result


@execute_node.register(ops.Reduction, SeriesGroupBy, type(None))
def execute_reduction_series_groupby(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(data, type(op).__name__.lower())


@execute_node.register(ops.First, SeriesGroupBy, type(None))
def execute_first_series_groupby(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(data, lambda x: getattr(x, "iat", x)[0])


@execute_node.register(ops.Last, SeriesGroupBy, type(None))
def execute_last_series_groupby(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(data, lambda x: getattr(x, "iat", x)[-1])


variance_ddof = {"pop": 0, "sample": 1}


@execute_node.register(ops.Variance, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_var(op, data, _, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "var", ddof=variance_ddof[op.how])


@execute_node.register(ops.StandardDev, SeriesGroupBy, type(None))
def execute_reduction_series_groupby_std(op, data, _, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "std", ddof=variance_ddof[op.how])


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    SeriesGroupBy,
    type(None),
)
def execute_count_distinct_series_groupby(op, data, _, aggcontext=None, **kwargs):
    return aggcontext.agg(data, "nunique")


@execute_node.register(ops.Arbitrary, SeriesGroupBy, type(None))
def execute_arbitrary_series_groupby(op, data, _, aggcontext=None, **kwargs):
    how = op.how
    if how is None:
        how = "first"

    if how not in {"first", "last"}:
        raise com.OperationNotDefinedError(f"Arbitrary {how!r} is not supported")
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
def execute_reduction_series_gb_mask(op, data, mask, aggcontext=None, **kwargs):
    method = operator.methodcaller(type(op).__name__.lower())
    return aggcontext.agg(
        data, functools.partial(_filtered_reduction, mask.obj, method)
    )


@execute_node.register(ops.First, SeriesGroupBy, SeriesGroupBy)
def execute_first_series_gb_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data, functools.partial(_filtered_reduction, mask.obj, lambda x: x.iloc[0])
    )


@execute_node.register(ops.Last, SeriesGroupBy, SeriesGroupBy)
def execute_last_series_gb_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data, functools.partial(_filtered_reduction, mask.obj, lambda x: x.iloc[-1])
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


@execute_node.register(ops.CountStar, DataFrameGroupBy, type(None))
def execute_count_star_frame_groupby(op, data, _, **kwargs):
    return data.size()


@execute_node.register(ops.CountDistinctStar, DataFrameGroupBy, type(None))
def execute_count_distinct_star_frame_groupby(op, data, _, **kwargs):
    return data.nunique()


@execute_node.register(ops.Reduction, pd.Series, (pd.Series, type(None)))
def execute_reduction_series_mask(op, data, mask, aggcontext=None, **kwargs):
    operand = data[mask] if mask is not None else data
    return aggcontext.agg(operand, type(op).__name__.lower())


@execute_node.register(ops.First, pd.Series, (pd.Series, type(None)))
def execute_first_series_mask(op, data, mask, aggcontext=None, **kwargs):
    operand = data[mask] if mask is not None else data

    def _first(x):
        return getattr(x, "iloc", x)[0]

    return aggcontext.agg(operand, _first)


@execute_node.register(ops.Last, pd.Series, (pd.Series, type(None)))
def execute_last_series_mask(op, data, mask, aggcontext=None, **kwargs):
    operand = data[mask] if mask is not None else data

    def _last(x):
        return getattr(x, "iloc", x)[-1]

    return aggcontext.agg(operand, _last)


@execute_node.register(
    (ops.CountDistinct, ops.ApproxCountDistinct),
    pd.Series,
    (pd.Series, type(None)),
)
def execute_count_distinct_series_mask(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(data[mask] if mask is not None else data, "nunique")


@execute_node.register(ops.Arbitrary, pd.Series, (pd.Series, type(None)))
def execute_arbitrary_series_mask(op, data, mask, aggcontext=None, **kwargs):
    if op.how == "first":
        index = 0
    elif op.how == "last":
        index = -1
    else:
        raise com.OperationNotDefinedError(f"Arbitrary {op.how!r} is not supported")

    data = data[mask] if mask is not None else data
    return data.iloc[index]


@execute_node.register(ops.StandardDev, pd.Series, (pd.Series, type(None)))
def execute_standard_dev_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        "std",
        ddof=variance_ddof[op.how],
    )


@execute_node.register(ops.Variance, pd.Series, (pd.Series, type(None)))
def execute_variance_series(op, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data,
        "var",
        ddof=variance_ddof[op.how],
    )


@execute_node.register((ops.Any, ops.All), pd.Series, (pd.Series, type(None)))
def execute_any_all_series(op, data, mask, aggcontext=None, **kwargs):
    if mask is not None:
        data = data.loc[mask]
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


@execute_node.register((ops.Any, ops.All), SeriesGroupBy, type(None))
def execute_any_all_series_group_by(op, data, mask, aggcontext=None, **kwargs):
    if mask is not None:
        data = data.obj.loc[mask].groupby(get_grouping(data.grouper.groupings))
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


@execute_node.register(ops.CountStar, pd.DataFrame, type(None))
def execute_count_star_frame(op, data, _, **kwargs):
    return len(data)


@execute_node.register(ops.CountStar, pd.DataFrame, pd.Series)
def execute_count_star_frame_filter(op, data, where, **kwargs):
    return len(data) - len(where) + where.sum()


@execute_node.register(ops.CountDistinctStar, pd.DataFrame, type(None))
def execute_count_distinct_star_frame(op, data, _, **kwargs):
    return len(data.drop_duplicates())


@execute_node.register(ops.CountDistinctStar, pd.DataFrame, pd.Series)
def execute_count_distinct_star_frame_filter(op, data, filt, **kwargs):
    return len(data.loc[filt].drop_duplicates())


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


@execute_node.register(ops.Mode, pd.Series, (pd.Series, type(None)))
def execute_mode_series(_, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data, lambda x: x.mode().iloc[0]
    )


@execute_node.register(ops.Mode, SeriesGroupBy, (SeriesGroupBy, type(None)))
def execute_mode_series_groupby(_, data, mask, aggcontext=None, **kwargs):
    def mode(x):
        return x.mode().iloc[0]

    if mask is not None:
        mode = functools.partial(_filtered_reduction, mask.obj, mode)

    return aggcontext.agg(data, mode)


@execute_node.register(ops.ApproxMedian, pd.Series, (pd.Series, type(None)))
def execute_approx_median_series(_, data, mask, aggcontext=None, **kwargs):
    return aggcontext.agg(
        data[mask] if mask is not None else data, lambda x: x.median()
    )


@execute_node.register(ops.ApproxMedian, SeriesGroupBy, (SeriesGroupBy, type(None)))
def execute_approx_median_series_groupby(_, data, mask, aggcontext=None, **kwargs):
    median = pd.Series.median

    if mask is not None:
        median = functools.partial(_filtered_reduction, mask.obj, median)

    return aggcontext.agg(data, median)


@execute_node.register((ops.Not, ops.Negate), (bool, np.bool_))
def execute_not_bool(_, data, **kwargs):
    return not data


def _execute_binary_op_impl(op, left, right, **_):
    op_type = type(op)
    try:
        operation = constants.BINARY_OPERATIONS[op_type]
    except KeyError:
        raise com.OperationNotDefinedError(
            f"Binary operation {op_type.__name__} not implemented"
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
@execute_node.register(ops.BitwiseBinary, integer_types, integer_types)
@execute_node.register(ops.BitwiseBinary, pd.Series, integer_types)
@execute_node.register(ops.BitwiseBinary, integer_types, pd.Series)
def execute_binary_op(op, left, right, **kwargs):
    return _execute_binary_op_impl(op, left, right, **kwargs)


@execute_node.register(ops.Comparison, pd.Series, date_types)
def execute_binary_op_date(op, left, right, **kwargs):
    return _execute_binary_op_impl(
        op, pd.to_datetime(left), pd.to_datetime(right), **kwargs
    )


@execute_node.register(ops.Binary, SeriesGroupBy, SeriesGroupBy)
def execute_binary_op_series_group_by(op, left, right, **kwargs):
    left_groupings = get_grouping(left.grouper.groupings)
    right_groupings = get_grouping(right.grouper.groupings)
    if left_groupings != right_groupings:
        raise ValueError(
            f"Cannot perform {type(op).__name__} operation on two series with "
            "different groupings"
        )
    result = execute_binary_op(op, left.obj, right.obj, **kwargs)
    return result.groupby(left_groupings, group_keys=False)


@execute_node.register(ops.Binary, SeriesGroupBy, simple_types)
def execute_binary_op_series_gb_simple(op, left, right, **kwargs):
    result = execute_binary_op(op, left.obj, right, **kwargs)
    return result.groupby(get_grouping(left.grouper.groupings), group_keys=False)


@execute_node.register(ops.Binary, simple_types, SeriesGroupBy)
def execute_binary_op_simple_series_gb(op, left, right, **kwargs):
    result = execute_binary_op(op, left, right.obj, **kwargs)
    return result.groupby(get_grouping(right.grouper.groupings), group_keys=False)


@execute_node.register(ops.Unary, SeriesGroupBy)
def execute_unary_op_series_gb(op, operand, **kwargs):
    result = execute_node(op, operand.obj, **kwargs)
    return result.groupby(get_grouping(operand.grouper.groupings), group_keys=False)


@execute_node.register(
    (ops.Log, ops.Round),
    SeriesGroupBy,
    (numbers.Real, decimal.Decimal, type(None)),
)
def execute_log_series_gb_others(op, left, right, **kwargs):
    result = execute_node(op, left.obj, right, **kwargs)
    return result.groupby(get_grouping(left.grouper.groupings), group_keys=False)


@execute_node.register((ops.Log, ops.Round), SeriesGroupBy, SeriesGroupBy)
def execute_log_series_gb_series_gb(op, left, right, **kwargs):
    result = execute_node(op, left.obj, right.obj, **kwargs)
    return result.groupby(get_grouping(left.grouper.groupings), group_keys=False)


@execute_node.register(ops.Not, pd.Series)
def execute_not_series(op, data, **kwargs):
    return ~data


@execute_node.register(ops.StringSplit, pd.Series, (pd.Series, str))
def execute_string_split(op, data, delimiter, **kwargs):
    # Doing the iteration using `map` is much faster than doing the iteration
    # using `Series.apply` due to Pandas-related overhead.
    return pd.Series(np.array(s.split(delimiter)) for s in data)


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
    merged = left.merge(right, on=list(left.columns), how="outer", indicator=True)
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


@execute_node.register(ops.StringConcat, tuple)
def execute_node_string_concat(op, values, **kwargs):
    values = [execute(arg, **kwargs) for arg in values]
    return functools.reduce(operator.add, values)


@execute_node.register(ops.StringJoin, collections.abc.Sequence)
def execute_node_string_join(op, args, **kwargs):
    return op.sep.join(args)


@execute_node.register(ops.InValues, object, tuple)
def execute_node_scalar_in_values(op, data, elements, **kwargs):
    elements = [execute(arg, **kwargs) for arg in elements]
    return data in elements


@execute_node.register(ops.InColumn, object, np.ndarray)
def execute_node_scalar_in_column(op, data, elements, **kwargs):
    return data in elements


@execute_node.register(ops.InValues, pd.Series, tuple)
def execute_node_column_in_values(op, data, elements, **kwargs):
    elements = [execute(arg, **kwargs) for arg in elements]
    return data.isin(elements)


@execute_node.register(ops.InColumn, pd.Series, pd.Series)
def execute_node_column_in_column(op, data, elements, **kwargs):
    return data.isin(elements)


@execute_node.register(ops.InValues, SeriesGroupBy, tuple)
def execute_node_group_in_values(op, data, elements, **kwargs):
    elements = [execute(arg, **kwargs) for arg in elements]
    return data.obj.isin(elements).groupby(
        get_grouping(data.grouper.groupings), group_keys=False
    )


@execute_node.register(ops.InColumn, SeriesGroupBy, pd.Series)
def execute_node_group_in_column(op, data, elements, **kwargs):
    return data.obj.isin(elements).groupby(
        get_grouping(data.grouper.groupings), group_keys=False
    )


def pd_where(cond, true, false):
    """Execute `where` following ibis's intended semantics."""
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


@execute_node.register(ops.IfElse, (pd.Series, *boolean_types), pd.Series, pd.Series)
@execute_node.register(ops.IfElse, (pd.Series, *boolean_types), pd.Series, simple_types)
@execute_node.register(ops.IfElse, (pd.Series, *boolean_types), simple_types, pd.Series)
@execute_node.register(ops.IfElse, (pd.Series, *boolean_types), type(None), type(None))
def execute_node_where(op, cond, true, false, **kwargs):
    return pd_where(cond, true, false)


# For true/false as scalars, we only support identical type pairs + None to
# limit the size of the dispatch table and not have to worry about type
# promotion.
for typ in (str, *scalar_types):
    for cond_typ in (pd.Series, *boolean_types):
        execute_node.register(ops.IfElse, cond_typ, typ, typ)(execute_node_where)
        execute_node.register(ops.IfElse, cond_typ, type(None), typ)(execute_node_where)
        execute_node.register(ops.IfElse, cond_typ, typ, type(None))(execute_node_where)


@execute_node.register(ops.DatabaseTable, PandasBackend)
def execute_database_table_client(
    op, client, timecontext: TimeContext | None, **kwargs
):
    df = client.dictionary[op.name]
    if timecontext:
        begin, end = timecontext
        time_col = get_time_col()
        if time_col not in df:
            raise com.IbisError(
                f"Table {op.name} must have a time column named {time_col}"
                " to execute with time context."
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


@execute_node.register(ops.DropNa, pd.DataFrame)
def execute_node_dropna_dataframe(op, df, **kwargs):
    if op.subset is not None:
        subset = [col.name for col in op.subset]
    else:
        subset = None
    return df.dropna(how=op.how, subset=subset)


@execute_node.register(ops.FillNa, pd.DataFrame, simple_types)
def execute_node_fillna_dataframe_scalar(op, df, replacements, **kwargs):
    return df.fillna(replacements)


@execute_node.register(ops.FillNa, pd.DataFrame)
def execute_node_fillna_dataframe_dict(op, df, **kwargs):
    return df.fillna(dict(op.replacements))


@execute_node.register(ops.NullIf, simple_types, simple_types)
def execute_node_nullif_scalars(op, value1, value2, **kwargs):
    return np.nan if value1 == value2 else value1


@execute_node.register(ops.NullIf, pd.Series, (pd.Series, *simple_types))
def execute_node_nullif_series(op, left, right, **kwargs):
    return left.where(left != right)


@execute_node.register(ops.NullIf, simple_types, pd.Series)
def execute_node_nullif_scalar_series(op, value, series, **kwargs):
    return series.where(series != value)


def coalesce(values):
    return functools.reduce(
        lambda a1, a2: np.where(pd.isnull(a1), a2, a1),
        values,
    )


@toolz.curry
def promote_to_sequence(length, obj):
    try:
        return obj.values
    except AttributeError:
        return np.repeat(obj, length)


def compute_row_reduction(func, values, **kwargs):
    final_sizes = {len(x) for x in values if isinstance(x, Sized)}
    if not final_sizes:
        return func(values)
    (final_size,) = final_sizes
    raw = func(list(map(promote_to_sequence(final_size), values)), **kwargs)
    return pd.Series(raw).squeeze()


@execute_node.register(ops.Greatest, tuple)
def execute_node_greatest_list(op, values, **kwargs):
    values = [execute(arg, **kwargs) for arg in values]
    return compute_row_reduction(np.maximum.reduce, values, axis=0)


@execute_node.register(ops.Least, tuple)
def execute_node_least_list(op, values, **kwargs):
    values = [execute(arg, **kwargs) for arg in values]
    return compute_row_reduction(np.minimum.reduce, values, axis=0)


@execute_node.register(ops.Coalesce, tuple)
def execute_node_coalesce(op, values, **kwargs):
    # TODO: this is slow
    values = [execute(arg, **kwargs) for arg in values]
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
        value = result.iloc[0]
        try:
            return value.item()
        except AttributeError:
            return value
    return result


def _build_select(op, whens, thens, otherwise, func=None, **kwargs):
    if func is None:
        func = lambda x: x

    whens_ = []
    grouped = 0
    for when in whens:
        res = execute(when, **kwargs)
        obj = getattr(res, "obj", res)
        grouped += obj is not res
        whens_.append(obj)

    thens_ = []
    for then in thens:
        res = execute(then, **kwargs)
        obj = getattr(res, "obj", res)
        grouped += obj is not res
        thens_.append(obj)

    if otherwise is None:
        otherwise = np.nan

    raw = np.select(func(whens_), thens_, otherwise)

    if grouped:
        return pd.Series(raw).groupby(get_grouping(res.grouper.groupings))
    return wrap_case_result(raw, op.to_expr())


@execute_node.register(ops.SearchedCase, tuple, tuple, object)
def execute_searched_case(op, whens, thens, otherwise, **kwargs):
    return _build_select(op, whens, thens, otherwise, **kwargs)


@execute_node.register(ops.SimpleCase, object, tuple, tuple, object)
def execute_simple_case_scalar(op, value, whens, thens, otherwise, **kwargs):
    value = getattr(value, "obj", value)
    return _build_select(
        op,
        whens,
        thens,
        otherwise,
        func=lambda whens: np.asarray(whens) == value,
        **kwargs,
    )


@execute_node.register(ops.SimpleCase, (pd.Series, SeriesGroupBy), tuple, tuple, object)
def execute_simple_case_series(op, value, whens, thens, otherwise, **kwargs):
    value = getattr(value, "obj", value)
    return _build_select(
        op,
        whens,
        thens,
        otherwise,
        func=lambda whens: [value == when for when in whens],
        **kwargs,
    )


@execute_node.register(ops.Distinct, pd.DataFrame)
def execute_distinct_dataframe(op, df, **kwargs):
    return df.drop_duplicates()


@execute_node.register(ops.TableArrayView, pd.DataFrame)
def execute_table_array_view(op, _, **kwargs):
    return execute(op.table).squeeze()


@execute_node.register(ops.InMemoryTable)
def execute_in_memory_table(op, **kwargs):
    return op.data.to_frame()


@execute_node.register(ops.Sample, pd.DataFrame, object, object)
def execute_sample(op, data, fraction, seed, **kwargs):
    return data.sample(frac=fraction, random_state=seed)
