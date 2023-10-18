from __future__ import annotations

import operator
from typing import Annotated, Union

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.patterns import As, Attrs
from ibis.common.temporal import DateUnit, IntervalUnit, TimestampUnit, TimeUnit
from ibis.expr.operations.core import Binary, Scalar, Unary, Value
from ibis.expr.operations.logical import Between


@public
class TemporalUnary(Unary):
    arg: Value[dt.Temporal]


@public
class TimestampUnary(Unary):
    arg: Value[dt.Timestamp]


@public
class TimestampTruncate(Value):
    arg: Value[dt.Timestamp]
    unit: IntervalUnit

    shape = rlz.shape_like("arg")
    dtype = dt.timestamp


@public
class DateTruncate(Value):
    arg: Value[dt.Date]
    unit: DateUnit

    shape = rlz.shape_like("arg")
    dtype = dt.date


@public
class TimeTruncate(Value):
    arg: Value[dt.Time]
    unit: TimeUnit

    shape = rlz.shape_like("arg")
    dtype = dt.time


@public
class TimestampBucket(Value):
    arg: Value[dt.Timestamp]
    interval: Scalar[dt.Interval]
    offset: Union[Scalar[dt.Interval], None] = None

    shape = rlz.shape_like("arg")
    dtype = dt.timestamp


@public
class Strftime(Value):
    arg: Value[dt.Temporal]
    format_str: Value[dt.String]

    shape = rlz.shape_like("arg")
    dtype = dt.string


@public
class StringToTimestamp(Value):
    arg: Value[dt.String]
    format_str: Value[dt.String]

    shape = rlz.shape_like("arg")
    dtype = dt.Timestamp(timezone="UTC")


@public
class ExtractTemporalField(TemporalUnary):
    dtype = dt.int32


@public
class ExtractDateField(ExtractTemporalField):
    arg: Value[dt.Date | dt.Timestamp]


@public
class ExtractTimeField(ExtractTemporalField):
    arg: Value[dt.Time | dt.Timestamp]


@public
class ExtractYear(ExtractDateField):
    pass


@public
class ExtractMonth(ExtractDateField):
    pass


@public
class ExtractDay(ExtractDateField):
    pass


@public
class ExtractDayOfYear(ExtractDateField):
    pass


@public
class ExtractQuarter(ExtractDateField):
    pass


@public
class ExtractEpochSeconds(ExtractDateField):
    pass


@public
class ExtractWeekOfYear(ExtractDateField):
    pass


@public
class ExtractHour(ExtractTimeField):
    pass


@public
class ExtractMinute(ExtractTimeField):
    pass


@public
class ExtractSecond(ExtractTimeField):
    pass


@public
class ExtractMicrosecond(ExtractTimeField):
    pass


@public
class ExtractMillisecond(ExtractTimeField):
    pass


@public
class DayOfWeekIndex(Unary):
    arg: Value[dt.Date | dt.Timestamp]

    dtype = dt.int16


@public
class DayOfWeekName(Unary):
    arg: Value[dt.Date | dt.Timestamp]

    dtype = dt.string


@public
class Time(Unary):
    dtype = dt.time


@public
class Date(Unary):
    dtype = dt.date


@public
class DateFromYMD(Value):
    year: Value[dt.Integer]
    month: Value[dt.Integer]
    day: Value[dt.Integer]

    dtype = dt.date
    shape = rlz.shape_like("args")


@public
class TimeFromHMS(Value):
    hours: Value[dt.Integer]
    minutes: Value[dt.Integer]
    seconds: Value[dt.Integer]

    dtype = dt.time
    shape = rlz.shape_like("args")


@public
class TimestampFromYMDHMS(Value):
    year: Value[dt.Integer]
    month: Value[dt.Integer]
    day: Value[dt.Integer]
    hours: Value[dt.Integer]
    minutes: Value[dt.Integer]
    seconds: Value[dt.Integer]

    dtype = dt.timestamp
    shape = rlz.shape_like("args")


@public
class TimestampFromUNIX(Value):
    arg: Value
    unit: TimestampUnit

    dtype = dt.timestamp
    shape = rlz.shape_like("arg")


TimeInterval = Annotated[dt.Interval, Attrs(unit=As(TimeUnit))]
DateInterval = Annotated[dt.Interval, Attrs(unit=As(DateUnit))]


@public
class DateAdd(Binary):
    left: Value[dt.Date]
    right: Value[DateInterval]

    dtype = rlz.dtype_like("left")


@public
class DateSub(Binary):
    left: Value[dt.Date]
    right: Value[DateInterval]

    dtype = rlz.dtype_like("left")


@public
class DateDiff(Binary):
    left: Value[dt.Date]
    right: Value[dt.Date]

    dtype = dt.Interval("D")


@public
class TimeAdd(Binary):
    left: Value[dt.Time]
    right: Value[TimeInterval]

    dtype = rlz.dtype_like("left")


@public
class TimeSub(Binary):
    left: Value[dt.Time]
    right: Value[TimeInterval]

    dtype = rlz.dtype_like("left")


@public
class TimeDiff(Binary):
    left: Value[dt.Time]
    right: Value[dt.Time]

    dtype = dt.Interval("s")


@public
class TimestampAdd(Binary):
    left: Value[dt.Timestamp]
    right: Value[dt.Interval]

    dtype = rlz.dtype_like("left")


@public
class TimestampSub(Binary):
    left: Value[dt.Timestamp]
    right: Value[dt.Interval]

    dtype = rlz.dtype_like("left")


@public
class TimestampDiff(Binary):
    left: Value[dt.Timestamp]
    right: Value[dt.Timestamp]

    dtype = dt.Interval("s")


@public
class IntervalBinary(Binary):
    @attribute
    def dtype(self):
        interval_unit_args = [
            arg.dtype.unit for arg in (self.left, self.right) if arg.dtype.is_interval()
        ]
        unit = rlz._promote_interval_resolution(interval_unit_args)

        return self.left.dtype.copy(unit=unit)


@public
class IntervalAdd(IntervalBinary):
    left: Value[dt.Interval]
    right: Value[dt.Interval]
    op = operator.add


@public
class IntervalSubtract(IntervalBinary):
    left: Value[dt.Interval]
    right: Value[dt.Interval]
    op = operator.sub


@public
class IntervalMultiply(IntervalBinary):
    left: Value[dt.Interval]
    right: Value[dt.Numeric | dt.Boolean]
    op = operator.mul


@public
class IntervalFloorDivide(IntervalBinary):
    left: Value[dt.Interval]
    right: Value[dt.Numeric | dt.Boolean]
    op = operator.floordiv


@public
class IntervalFromInteger(Value):
    arg: Value[dt.Integer]
    unit: IntervalUnit

    shape = rlz.shape_like("arg")

    @attribute
    def dtype(self):
        return dt.Interval(self.unit)

    @property
    def resolution(self):
        return self.dtype.resolution


@public
class BetweenTime(Between):
    arg: Value[dt.Time | dt.Timestamp]
    lower_bound: Value[dt.Time | dt.String]
    upper_bound: Value[dt.Time | dt.String]


class TemporalDelta(Value):
    part: Value[dt.String]
    shape = rlz.shape_like("args")
    dtype = dt.int64


@public
class TimeDelta(TemporalDelta):
    left: Value[dt.Time]
    right: Value[dt.Time]


@public
class DateDelta(TemporalDelta):
    left: Value[dt.Date]
    right: Value[dt.Date]


@public
class TimestampDelta(TemporalDelta):
    left: Value[dt.Timestamp]
    right: Value[dt.Timestamp]


public(ExtractTimestampField=ExtractTemporalField)
