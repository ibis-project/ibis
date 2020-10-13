import datetime

import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.dataframe.groupby import SeriesGroupBy
from pandas import Timedelta, Timestamp, to_datetime
from pandas.tseries.offsets import DateOffset

import ibis
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


@execute_node.register(ops.Strftime, Timestamp, str)
def execute_strftime_timestamp_str(op, data, format_string, **kwargs):
    return data.strftime(format_string)


@execute_node.register(ops.Strftime, dd.Series, str)
def execute_strftime_series_str(op, data, format_string, **kwargs):
    return data.dt.strftime(format_string)


@execute_node.register(ops.ExtractTemporalField, Timestamp)
def execute_extract_timestamp_field_timestamp(op, data, **kwargs):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data, field_name)


@execute_node.register(ops.ExtractTemporalField, dd.Series)
def execute_extract_timestamp_field_series(op, data, **kwargs):
    field_name = type(op).__name__.lower().replace('extract', '')
    return getattr(data.dt, field_name).astype(np.int32)


@execute_node.register(ops.ExtractMillisecond, Timestamp)
def execute_extract_millisecond_timestamp(op, data, **kwargs):
    return int(data.microsecond // 1000.0)


@execute_node.register(ops.ExtractMillisecond, dd.Series)
def execute_extract_millisecond_series(op, data, **kwargs):
    return (data.dt.microsecond // 1000).astype(np.int32)


@execute_node.register(ops.ExtractEpochSeconds, (Timestamp, dd.Series))
def execute_epoch_seconds(op, data, **kwargs):
    return data.astype('int64') // int(1e9)


@execute_node.register(
    ops.BetweenTime,
    dd.Series,
    (dd.Series, str, datetime.time),
    (dd.Series, str, datetime.time),
)
def execute_between_time(op, data, lower, upper, **kwargs):
    # TODO - This can be done better. Not sure we handle types correctly either
    indexer = (
        (data.dt.time.astype(str) >= lower)
        & (data.dt.time.astype(str) <= upper)
    ).to_dask_array(True)

    result = da.zeros(len(data), dtype=np.bool_)
    result[indexer] = True
    return dd.from_array(result)


@execute_node.register(ops.Date, dd.Series)
def execute_timestamp_date(op, data, **kwargs):
    return data.dt.floor('d')


@execute_node.register((ops.TimestampTruncate, ops.DateTruncate), dd.Series)
def execute_timestamp_truncate(op, data, **kwargs):
    dtype = 'datetime64[{}]'.format(op.unit)
    array = data.values.astype(dtype)
    return dd.Series(array, name=data.name)


OFFSET_CLASS = {
    "Y": DateOffset,
    "Q": DateOffset,
    "M": DateOffset,
    "W": DateOffset,
    # all other units are timedelta64s
}


@execute_node.register(ops.IntervalFromInteger, dd.Series)
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
        return Timedelta(data, unit=unit)
    return cls(**{resolution: data})


@execute_node.register(ops.Cast, dd.Series, dt.Interval)
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
        return Timedelta(data, unit=unit)
    return cls(**{resolution: data})


@execute_node.register(ops.TimestampAdd, timestamp_types, timedelta_types)
def execute_timestamp_add_datetime_timedelta(op, left, right, **kwargs):
    return Timestamp(left) + Timedelta(right)


@execute_node.register(ops.TimestampAdd, timestamp_types, dd.Series)
def execute_timestamp_add_datetime_series(op, left, right, **kwargs):
    return Timestamp(left) + right


@execute_node.register(ops.IntervalAdd, timedelta_types, timedelta_types)
def execute_interval_add_delta_delta(op, left, right, **kwargs):
    return op.op(Timedelta(left), Timedelta(right))


@execute_node.register(ops.IntervalAdd, timedelta_types, dd.Series)
@execute_node.register(
    ops.IntervalMultiply, timedelta_types, numeric_types + (dd.Series,)
)
def execute_interval_add_multiply_delta_series(op, left, right, **kwargs):
    return op.op(Timedelta(left), right)


@execute_node.register(
    (ops.TimestampAdd, ops.IntervalAdd), dd.Series, timedelta_types
)
def execute_timestamp_interval_add_series_delta(op, left, right, **kwargs):
    return left + Timedelta(right)


@execute_node.register(
    (ops.TimestampAdd, ops.IntervalAdd), dd.Series, dd.Series
)
def execute_timestamp_interval_add_series_series(op, left, right, **kwargs):
    return left + right


@execute_node.register(ops.TimestampSub, timestamp_types, timedelta_types)
def execute_timestamp_sub_datetime_timedelta(op, left, right, **kwargs):
    return Timestamp(left) - Timedelta(right)


@execute_node.register(
    (ops.TimestampDiff, ops.TimestampSub), timestamp_types, dd.Series
)
def execute_timestamp_diff_sub_datetime_series(op, left, right, **kwargs):
    return Timestamp(left) - right


@execute_node.register(ops.TimestampSub, dd.Series, timedelta_types)
def execute_timestamp_sub_series_timedelta(op, left, right, **kwargs):
    return left - Timedelta(right)


@execute_node.register(
    (ops.TimestampDiff, ops.TimestampSub, ops.IntervalSubtract),
    dd.Series,
    dd.Series,
)
def execute_timestamp_diff_sub_series_series(op, left, right, **kwargs):
    return left - right


@execute_node.register(ops.TimestampDiff, timestamp_types, timestamp_types)
def execute_timestamp_diff_datetime_datetime(op, left, right, **kwargs):
    return Timestamp(left) - Timestamp(right)


@execute_node.register(ops.TimestampDiff, dd.Series, timestamp_types)
def execute_timestamp_diff_series_datetime(op, left, right, **kwargs):
    return left - Timestamp(right)


@execute_node.register(
    ops.IntervalMultiply, dd.Series, numeric_types + (dd.Series,)
)
@execute_node.register(
    ops.IntervalFloorDivide,
    (Timedelta, dd.Series),
    numeric_types + (dd.Series,),
)
def execute_interval_multiply_fdiv_series_numeric(op, left, right, **kwargs):
    return op.op(left, right)


@execute_node.register(ops.TimestampFromUNIX, (dd.Series,) + integer_types)
def execute_timestamp_from_unix(op, data, **kwargs):
    return to_datetime(data, unit=op.unit)


@pre_execute.register(ops.TimestampNow)
@pre_execute.register(ops.TimestampNow, ibis.client.Client)
def pre_execute_timestamp_now(op, *args, **kwargs):
    timecontext = kwargs.get('timecontext', None)
    return Scope({op: Timestamp('now')}, timecontext)


@execute_node.register(ops.DayOfWeekIndex, (str, datetime.date))
def execute_day_of_week_index_any(op, value, **kwargs):
    return Timestamp(value).dayofweek


@execute_node.register(ops.DayOfWeekIndex, dd.Series)
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
    obj : Union[Series, Timestamp]

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
    return day_name(Timestamp(value))


@execute_node.register(ops.DayOfWeekName, dd.Series)
def execute_day_of_week_name_series(op, data, **kwargs):
    return day_name(data.dt)


@execute_node.register(ops.DayOfWeekName, SeriesGroupBy)
def execute_day_of_week_name_series_group_by(op, data, **kwargs):
    return day_name(data.obj.dt).groupby(data.grouper.groupings)


@execute_node.register(ops.DateSub, date_types, timedelta_types)
@execute_node.register((ops.DateDiff, ops.DateSub), date_types, dd.Series)
@execute_node.register(ops.DateSub, dd.Series, timedelta_types)
@execute_node.register((ops.DateDiff, ops.DateSub), dd.Series, dd.Series)
@execute_node.register(ops.DateDiff, date_types, date_types)
@execute_node.register(ops.DateDiff, dd.Series, date_types)
def execute_date_sub_diff(op, left, right, **kwargs):
    return left - right


@execute_node.register(ops.DateAdd, dd.Series, timedelta_types)
@execute_node.register(ops.DateAdd, timedelta_types, dd.Series)
@execute_node.register(ops.DateAdd, dd.Series, dd.Series)
@execute_node.register(ops.DateAdd, date_types, timedelta_types)
@execute_node.register(ops.DateAdd, timedelta_types, date_types)
@execute_node.register(ops.DateAdd, date_types, dd.Series)
@execute_node.register(ops.DateAdd, dd.Series, date_types)
def execute_date_add(op, left, right, **kwargs):
    return left + right
