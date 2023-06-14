from __future__ import annotations

import operator

from public import public
from typing_extensions import Annotated

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.patterns import As, Attrs
from ibis.common.temporal import DateUnit, IntervalUnit, TimestampUnit, TimeUnit
from ibis.expr.operations.core import Binary, Unary, Value
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

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.timestamp


@public
class DateTruncate(Value):
    arg: Value[dt.Date]
    unit: DateUnit

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.date


@public
class TimeTruncate(Value):
    arg: Value[dt.Time]
    unit: TimeUnit

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.time


@public
class Strftime(Value):
    arg: Value[dt.Temporal]
    format_str: Value[dt.String]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringToTimestamp(Value):
    arg: Value[dt.String]
    format_str: Value[dt.String]

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.Timestamp(timezone='UTC')


@public
class ExtractTemporalField(TemporalUnary):
    output_dtype = dt.int32


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

    output_dtype = dt.int16


@public
class DayOfWeekName(Unary):
    arg: Value[dt.Date | dt.Timestamp]

    output_dtype = dt.string


@public
class Time(Unary):
    output_dtype = dt.time


@public
class Date(Unary):
    output_dtype = dt.date


@public
class DateFromYMD(Value):
    year: Value[dt.Integer]
    month: Value[dt.Integer]
    day: Value[dt.Integer]

    output_dtype = dt.date
    output_shape = rlz.shape_like("args")


@public
class TimeFromHMS(Value):
    hours: Value[dt.Integer]
    minutes: Value[dt.Integer]
    seconds: Value[dt.Integer]

    output_dtype = dt.time
    output_shape = rlz.shape_like("args")


@public
class TimestampFromYMDHMS(Value):
    year: Value[dt.Integer]
    month: Value[dt.Integer]
    day: Value[dt.Integer]
    hours: Value[dt.Integer]
    minutes: Value[dt.Integer]
    seconds: Value[dt.Integer]

    output_dtype = dt.timestamp
    output_shape = rlz.shape_like("args")


@public
class TimestampFromUNIX(Value):
    arg: Value
    unit: TimestampUnit

    output_dtype = dt.timestamp
    output_shape = rlz.shape_like('arg')


TimeInterval = Annotated[dt.Interval, Attrs(unit=As(TimeUnit))]
DateInterval = Annotated[dt.Interval, Attrs(unit=As(DateUnit))]


@public
class DateAdd(Binary):
    left: Value[dt.Date]
    right: Value[DateInterval]

    output_dtype = rlz.dtype_like('left')


@public
class DateSub(Binary):
    left: Value[dt.Date]
    right: Value[DateInterval]

    output_dtype = rlz.dtype_like('left')


@public
class DateDiff(Binary):
    left: Value[dt.Date]
    right: Value[dt.Date]

    output_dtype = dt.Interval('D')


@public
class TimeAdd(Binary):
    left: Value[dt.Time]
    right: Value[TimeInterval]

    output_dtype = rlz.dtype_like('left')


@public
class TimeSub(Binary):
    left: Value[dt.Time]
    right: Value[TimeInterval]

    output_dtype = rlz.dtype_like('left')


@public
class TimeDiff(Binary):
    left: Value[dt.Time]
    right: Value[dt.Time]

    output_dtype = dt.Interval('s')


@public
class TimestampAdd(Binary):
    left: Value[dt.Timestamp]
    right: Value[dt.Interval]

    output_dtype = rlz.dtype_like('left')


@public
class TimestampSub(Binary):
    left: Value[dt.Timestamp]
    right: Value[dt.Interval]

    output_dtype = rlz.dtype_like('left')


@public
class TimestampDiff(Binary):
    left: Value[dt.Timestamp]
    right: Value[dt.Timestamp]

    output_dtype = dt.Interval('s')


@public
class IntervalBinary(Binary):
    @attribute.default
    def output_dtype(self):
        interval_unit_args = [
            arg.output_dtype.unit
            for arg in (self.left, self.right)
            if arg.output_dtype.is_interval()
        ]
        unit = rlz._promote_interval_resolution(interval_unit_args)

        return self.left.output_dtype.copy(unit=unit)


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

    output_shape = rlz.shape_like("arg")

    @attribute.default
    def output_dtype(self):
        return dt.Interval(self.unit)

    @property
    def resolution(self):
        return self.output_dtype.resolution


@public
class BetweenTime(Between):
    arg: Value[dt.Time | dt.Timestamp]
    lower_bound: Value[dt.Time | dt.String]
    upper_bound: Value[dt.Time | dt.String]


public(ExtractTimestampField=ExtractTemporalField)
