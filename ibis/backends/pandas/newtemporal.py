from __future__ import annotations

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import elementwise, rowwise, serieswise


def _timestamp_truncate(arg, unit):
    unit = {"m": "Min", "ms": "L"}.get(unit.short, unit.short)
    try:
        return arg.dt.floor(unit)
    except ValueError:
        return arg.dt.to_period(unit).dt.to_timestamp()


_serieswise_functions = {
    ops.ExtractYear: lambda arg: arg.dt.year,
    ops.ExtractQuarter: lambda arg: arg.dt.quarter,
    ops.ExtractMonth: lambda arg: arg.dt.month,
    ops.ExtractWeekOfYear: lambda arg: arg.dt.isocalendar().week.astype("int32"),
    ops.ExtractDay: lambda arg: arg.dt.day,
    ops.ExtractDayOfYear: lambda arg: arg.dt.dayofyear,
    ops.DayOfWeekIndex: lambda arg: pd.to_datetime(arg).dt.dayofweek,
    ops.DayOfWeekName: lambda arg: pd.to_datetime(arg).dt.day_name(),
    ops.ExtractHour: lambda arg: arg.dt.hour,
    ops.ExtractMinute: lambda arg: arg.dt.minute,
    ops.ExtractSecond: lambda arg: arg.dt.second,
    ops.ExtractEpochSeconds: lambda arg: arg.astype("datetime64[s]")
    .astype("int64")
    .astype("int32"),
    ops.ExtractMillisecond: lambda arg: arg.dt.microsecond // 1000,
    ops.ExtractMicrosecond: lambda arg: arg.dt.microsecond,
    ops.TimestampTruncate: _timestamp_truncate,
    ops.DateTruncate: _timestamp_truncate,
    ops.TimestampFromUNIX: lambda arg, unit: pd.to_datetime(arg, unit=unit.short),
    ops.Time: lambda arg: arg.dt.time,
}


@execute.register(ops.ExtractYear)
@execute.register(ops.ExtractQuarter)
@execute.register(ops.ExtractMonth)
@execute.register(ops.ExtractWeekOfYear)
@execute.register(ops.ExtractDay)
@execute.register(ops.ExtractDayOfYear)
@execute.register(ops.DayOfWeekIndex)
@execute.register(ops.DayOfWeekName)
@execute.register(ops.ExtractHour)
@execute.register(ops.ExtractMinute)
@execute.register(ops.ExtractSecond)
@execute.register(ops.ExtractEpochSeconds)
@execute.register(ops.ExtractMillisecond)
@execute.register(ops.ExtractMicrosecond)
@execute.register(ops.Strftime)
@execute.register(ops.TimestampTruncate)
@execute.register(ops.DateTruncate)
@execute.register(ops.TimestampFromUNIX)
@execute.register(ops.Time)
def execute_serieswise(op, **kwargs):
    func = _serieswise_functions[type(op)]
    return serieswise(func, **kwargs)


@execute.register(ops.DateAdd)
@execute.register(ops.TimestampAdd)
@execute.register(ops.IntervalAdd)
def execute_timestamp_add(op, left, right):
    return left + right


@execute.register(ops.DateSub)
@execute.register(ops.TimestampSub)
@execute.register(ops.DateDiff)
@execute.register(ops.IntervalSubtract)
@execute.register(ops.TimestampDiff)
def execute_timestamp_sub(op, left, right):
    return left - right


@execute.register(ops.IntervalMultiply)
def execute_interval_multiply(op, left, right):
    return left * right


@execute.register(ops.IntervalFloorDivide)
def execute_interval_floor_divide(op, left, right):
    return left // right


@execute.register(ops.BetweenTime)
def execute_between_time(op, arg, lower_bound, upper_bound):
    idx = pd.DatetimeIndex(arg)
    if idx.tz is not None:
        idx = idx.tz_convert(None)  # make naive because times are naive
    indexer = idx.indexer_between_time(lower_bound, upper_bound)
    result = np.zeros(len(arg), dtype=np.bool_)
    result[indexer] = True
    return pd.Series(result)


@execute.register(ops.IntervalFromInteger)
def execute_interval_from_integer(op, unit, **kwargs):
    if unit.short in {"Y", "Q", "M", "W"}:
        return elementwise(lambda v: pd.DateOffset(**{unit.plural: v}), **kwargs)
    else:
        return serieswise(
            lambda arg: arg.astype(f"timedelta64[{unit.short}]"), **kwargs
        )


@execute.register(ops.Strftime)
def execute_strftime(op, arg, format_str):
    if isinstance(format_str, pd.Series):
        data = {"arg": arg, "format_str": format_str}
        return rowwise(lambda row: row["arg"].strftime(row["format_str"]), data)
    else:
        return serieswise(
            lambda arg, format_str: arg.dt.strftime(format_str),
            arg=arg,
            format_str=format_str,
        )


@execute.register(ops.Date)
def execute_date(op, arg):
    return arg.dt.floor("d")


@execute.register(ops.TimestampNow)
def execute_timestamp_now(op):
    # timecontext = kwargs.get("timecontext", None)
    return pd.Timestamp("now", tz="UTC").tz_localize(None)
