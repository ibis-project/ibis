import datetime

import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis.client
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.scope import Scope

from ..core import (
    date_types,
    integer_types,
    numeric_types,
    timedelta_types,
    timestamp_types,
)
from ..dispatch import execute_node, pre_execute


@execute_node.register(ops.Strftime, pd.Timestamp, str)
def execute_strftime_timestamp_str(op, data, format_string, **kwargs):
    return data.strftime(format_string)


@execute_node.register(ops.Strftime, pd.Series, str)
def execute_strftime_series_str(op, data, format_string, **kwargs):
    return data.dt.strftime(format_string)


@execute_node.register(ops.ExtractTemporalField, pd.Timestamp)
def execute_extract_timestamp_field_timestamp(op, data, **kwargs):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data, field_name)


@execute_node.register(ops.ExtractTemporalField, pd.Series)
def execute_extract_timestamp_field_series(op, data, **kwargs):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data.dt, field_name).astype(np.int32)


@execute_node.register(ops.ExtractMillisecond, pd.Timestamp)
def execute_extract_millisecond_timestamp(op, data, **kwargs):
    return int(data.microsecond // 1000.0)


@execute_node.register(ops.ExtractMillisecond, pd.Series)
def execute_extract_millisecond_series(op, data, **kwargs):
    return (data.dt.microsecond // 1000).astype(np.int32)


@execute_node.register(ops.ExtractEpochSeconds, (pd.Timestamp, pd.Series))
def execute_epoch_seconds(op, data, **kwargs):
    return data.astype('int64') // int(1e9)


@execute_node.register(
    ops.BetweenTime,
    pd.Series,
    (pd.Series, str, datetime.time),
    (pd.Series, str, datetime.time),
)
def execute_between_time(op, data, lower, upper, **kwargs):
    indexer = pd.DatetimeIndex(data).indexer_between_time(lower, upper)
    result = np.zeros(len(data), dtype=np.bool_)
    result[indexer] = True
    return pd.Series(result)


@execute_node.register(ops.Date, pd.Series)
def execute_timestamp_date(op, data, **kwargs):
    return data.dt.floor('d')


@execute_node.register((ops.TimestampTruncate, ops.DateTruncate), pd.Series)
def execute_timestamp_truncate(op, data, **kwargs):
    dtype = 'datetime64[{}]'.format(op.unit)
    array = data.values.astype(dtype)
    return pd.Series(array, name=data.name)


OFFSET_CLASS = {
    "Y": pd.offsets.DateOffset,
    "Q": pd.offsets.DateOffset,
    "M": pd.offsets.DateOffset,
    "W": pd.offsets.DateOffset,
    # all other units are timedelta64s
}


@execute_node.register(ops.IntervalFromInteger, pd.Series)
def execute_interval_from_integer_series(op, data, **kwargs):
    unit = op.unit
    resolution = "{}s".format(op.resolution)
    cls = OFFSET_CLASS.get(unit, None)

    # fast path for timedelta conversion
    if cls is None:
        return data.astype("timedelta64[{}]".format(unit))
    return data.apply(
        lambda n, cls=cls, resolution=resolution: cls(**{resolution: n})
    )


@execute_node.register(ops.IntervalFromInteger, integer_types)
def execute_interval_from_integer_integer_types(op, data, **kwargs):
    unit = op.unit
    resolution = "{}s".format(op.resolution)
    cls = OFFSET_CLASS.get(unit, None)

    if cls is None:
        return pd.Timedelta(data, unit=unit)
    return cls(**{resolution: data})


@execute_node.register(ops.Cast, pd.Series, dt.Interval)
def execute_cast_integer_to_interval_series(op, data, type, **kwargs):
    to = op.to
    unit = to.unit
    resolution = "{}s".format(to.resolution)
    cls = OFFSET_CLASS.get(unit, None)

    if cls is None:
        return data.astype("timedelta64[{}]".format(unit))
    return data.apply(
        lambda n, cls=cls, resolution=resolution: cls(**{resolution: n})
    )


@execute_node.register(ops.Cast, integer_types, dt.Interval)
def execute_cast_integer_to_interval_integer_types(op, data, type, **kwargs):
    to = op.to
    unit = to.unit
    resolution = "{}s".format(to.resolution)
    cls = OFFSET_CLASS.get(unit, None)

    if cls is None:
        return pd.Timedelta(data, unit=unit)
    return cls(**{resolution: data})


@execute_node.register(ops.TimestampAdd, timestamp_types, timedelta_types)
def execute_timestamp_add_datetime_timedelta(op, left, right, **kwargs):
    return pd.Timestamp(left) + pd.Timedelta(right)


@execute_node.register(ops.TimestampAdd, timestamp_types, pd.Series)
def execute_timestamp_add_datetime_series(op, left, right, **kwargs):
    return pd.Timestamp(left) + right


@execute_node.register(ops.IntervalAdd, timedelta_types, timedelta_types)
def execute_interval_add_delta_delta(op, left, right, **kwargs):
    return op.op(pd.Timedelta(left), pd.Timedelta(right))


@execute_node.register(ops.IntervalAdd, timedelta_types, pd.Series)
@execute_node.register(
    ops.IntervalMultiply, timedelta_types, numeric_types + (pd.Series,)
)
def execute_interval_add_multiply_delta_series(op, left, right, **kwargs):
    return op.op(pd.Timedelta(left), right)


@execute_node.register(
    (ops.TimestampAdd, ops.IntervalAdd), pd.Series, timedelta_types
)
def execute_timestamp_interval_add_series_delta(op, left, right, **kwargs):
    return left + pd.Timedelta(right)


@execute_node.register(
    (ops.TimestampAdd, ops.IntervalAdd), pd.Series, pd.Series
)
def execute_timestamp_interval_add_series_series(op, left, right, **kwargs):
    return left + right


@execute_node.register(ops.TimestampSub, timestamp_types, timedelta_types)
def execute_timestamp_sub_datetime_timedelta(op, left, right, **kwargs):
    return pd.Timestamp(left) - pd.Timedelta(right)


@execute_node.register(
    (ops.TimestampDiff, ops.TimestampSub), timestamp_types, pd.Series
)
def execute_timestamp_diff_sub_datetime_series(op, left, right, **kwargs):
    return pd.Timestamp(left) - right


@execute_node.register(ops.TimestampSub, pd.Series, timedelta_types)
def execute_timestamp_sub_series_timedelta(op, left, right, **kwargs):
    return left - pd.Timedelta(right)


@execute_node.register(
    (ops.TimestampDiff, ops.TimestampSub, ops.IntervalSubtract),
    pd.Series,
    pd.Series,
)
def execute_timestamp_diff_sub_series_series(op, left, right, **kwargs):
    return left - right


@execute_node.register(ops.TimestampDiff, timestamp_types, timestamp_types)
def execute_timestamp_diff_datetime_datetime(op, left, right, **kwargs):
    return pd.Timestamp(left) - pd.Timestamp(right)


@execute_node.register(ops.TimestampDiff, pd.Series, timestamp_types)
def execute_timestamp_diff_series_datetime(op, left, right, **kwargs):
    return left - pd.Timestamp(right)


@execute_node.register(
    ops.IntervalMultiply, pd.Series, numeric_types + (pd.Series,)
)
@execute_node.register(
    ops.IntervalFloorDivide,
    (pd.Timedelta, pd.Series),
    numeric_types + (pd.Series,),
)
def execute_interval_multiply_fdiv_series_numeric(op, left, right, **kwargs):
    return op.op(left, right)


@execute_node.register(ops.TimestampFromUNIX, (pd.Series,) + integer_types)
def execute_timestamp_from_unix(op, data, **kwargs):
    return pd.to_datetime(data, unit=op.unit)


@pre_execute.register(ops.TimestampNow)
@pre_execute.register(ops.TimestampNow, ibis.client.Client)
def pre_execute_timestamp_now(op, *args, **kwargs):
    timecontext = kwargs.get('timecontext', None)
    return Scope({op: pd.Timestamp('now')}, timecontext)


@execute_node.register(ops.DayOfWeekIndex, (str, datetime.date))
def execute_day_of_week_index_any(op, value, **kwargs):
    return pd.Timestamp(value).dayofweek


@execute_node.register(ops.DayOfWeekIndex, pd.Series)
def execute_day_of_week_index_series(op, data, **kwargs):
    return data.dt.dayofweek.astype(np.int16)


@execute_node.register(ops.DayOfWeekIndex, SeriesGroupBy)
def execute_day_of_week_index_series_group_by(op, data, **kwargs):
    groupings = data.grouper.groupings
    return data.obj.dt.dayofweek.astype(np.int16).groupby(groupings)


def day_name(obj):
    """Backwards compatible name of day getting function.

    Parameters
    ----------
    obj : Union[Series, pd.Timestamp]

    Returns
    -------
    str
        The name of the day corresponding to `obj`
    """
    try:
        return obj.day_name()
    except AttributeError:
        return obj.weekday_name


@execute_node.register(ops.DayOfWeekName, (str, datetime.date))
def execute_day_of_week_name_any(op, value, **kwargs):
    return day_name(pd.Timestamp(value))


@execute_node.register(ops.DayOfWeekName, pd.Series)
def execute_day_of_week_name_series(op, data, **kwargs):
    return day_name(data.dt)


@execute_node.register(ops.DayOfWeekName, SeriesGroupBy)
def execute_day_of_week_name_series_group_by(op, data, **kwargs):
    return day_name(data.obj.dt).groupby(data.grouper.groupings)


@execute_node.register(ops.DateSub, date_types, timedelta_types)
@execute_node.register((ops.DateDiff, ops.DateSub), date_types, pd.Series)
@execute_node.register(ops.DateSub, pd.Series, timedelta_types)
@execute_node.register((ops.DateDiff, ops.DateSub), pd.Series, pd.Series)
@execute_node.register(ops.DateDiff, date_types, date_types)
@execute_node.register(ops.DateDiff, pd.Series, date_types)
def execute_date_sub_diff(op, left, right, **kwargs):
    return left - right


@execute_node.register(ops.DateAdd, pd.Series, timedelta_types)
@execute_node.register(ops.DateAdd, timedelta_types, pd.Series)
@execute_node.register(ops.DateAdd, pd.Series, pd.Series)
@execute_node.register(ops.DateAdd, date_types, timedelta_types)
@execute_node.register(ops.DateAdd, timedelta_types, date_types)
@execute_node.register(ops.DateAdd, date_types, pd.Series)
@execute_node.register(ops.DateAdd, pd.Series, date_types)
def execute_date_add(op, left, right, **kwargs):
    return left + right
