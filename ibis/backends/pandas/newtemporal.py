from __future__ import annotations

import numpy as np
import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.newpandas import execute
from ibis.backends.pandas.newutils import columnwise, elementwise, rowwise, serieswise


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
@execute.register(ops.TimestampTruncate)
@execute.register(ops.DateTruncate)
@execute.register(ops.TimestampFromUNIX)
def execute_columnwise(op, arg, **kwargs):
    func = _serieswise_functions[type(op)]
    return serieswise(func, arg, **kwargs)


@execute.register(ops.DateAdd)
@execute.register(ops.TimestampAdd)
def execute_timestamp_add(op, left, right):
    return left + right


@execute.register(ops.DateSub)
@execute.register(ops.TimestampSub)
def execute_timestamp_sub(op, left, right):
    return left - right


@execute.register(ops.IntervalFromInteger)
def execute_interval_from_integer(op, arg, unit):
    if unit.short in {"Y", "Q", "M", "W"}:
        return elementwise(lambda v: pd.DateOffset(**{unit.plural: v}), arg)
    else:
        return serieswise(lambda arg: arg.astype(f"timedelta64[{unit.short}]"), arg)
