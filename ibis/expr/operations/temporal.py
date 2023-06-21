from __future__ import annotations

import operator

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.temporal import DateUnit, IntervalUnit, TimestampUnit, TimeUnit
from ibis.expr.operations.core import Binary, Unary, Value
from ibis.expr.operations.logical import Between


@public
class TemporalUnary(Unary):
    arg = rlz.temporal


@public
class TimestampUnary(Unary):
    arg = rlz.timestamp


@public
class TimestampTruncate(Value):
    arg = rlz.timestamp
    unit = rlz.coerced_to(IntervalUnit)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.timestamp


@public
class DateTruncate(Value):
    arg = rlz.date
    unit = rlz.coerced_to(DateUnit)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.date


@public
class TimeTruncate(Value):
    arg = rlz.time
    unit = rlz.coerced_to(TimeUnit)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.time


@public
class Strftime(Value):
    arg = rlz.temporal
    format_str = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.string


@public
class StringToTimestamp(Value):
    arg = rlz.string
    format_str = rlz.string

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.Timestamp(timezone='UTC')


@public
class ExtractTemporalField(TemporalUnary):
    output_dtype = dt.int32


@public
class ExtractDateField(ExtractTemporalField):
    arg = rlz.one_of([rlz.date, rlz.timestamp])


@public
class ExtractTimeField(ExtractTemporalField):
    arg = rlz.one_of([rlz.time, rlz.timestamp])


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
    arg = rlz.one_of([rlz.date, rlz.timestamp])
    output_dtype = dt.int16


@public
class DayOfWeekName(Unary):
    arg = rlz.one_of([rlz.date, rlz.timestamp])
    output_dtype = dt.string


@public
class Time(Unary):
    output_dtype = dt.time


@public
class Date(Unary):
    output_dtype = dt.date


@public
class DateFromYMD(Value):
    year = rlz.integer
    month = rlz.integer
    day = rlz.integer

    output_dtype = dt.date
    output_shape = rlz.shape_like("args")


@public
class TimeFromHMS(Value):
    hours = rlz.integer
    minutes = rlz.integer
    seconds = rlz.integer

    output_dtype = dt.time
    output_shape = rlz.shape_like("args")


@public
class TimestampFromYMDHMS(Value):
    year = rlz.integer
    month = rlz.integer
    day = rlz.integer
    hours = rlz.integer
    minutes = rlz.integer
    seconds = rlz.integer

    output_dtype = dt.timestamp
    output_shape = rlz.shape_like("args")


@public
class TimestampFromUNIX(Value):
    arg = rlz.any
    unit = rlz.coerced_to(TimestampUnit)

    output_dtype = dt.timestamp
    output_shape = rlz.shape_like('arg')


@public
class DateAdd(Binary):
    left = rlz.date
    right = rlz.interval(units={'Y', 'Q', 'M', 'W', 'D'})
    output_dtype = rlz.dtype_like('left')


@public
class DateSub(Binary):
    left = rlz.date
    right = rlz.interval(units={'Y', 'Q', 'M', 'W', 'D'})
    output_dtype = rlz.dtype_like('left')


@public
class DateDiff(Binary):
    left = rlz.date
    right = rlz.date
    output_dtype = dt.Interval('D')


@public
class TimeAdd(Binary):
    left = rlz.time
    right = rlz.interval(units={'h', 'm', 's', 'ms', 'us', 'ns'})
    output_dtype = rlz.dtype_like('left')


@public
class TimeSub(Binary):
    left = rlz.time
    right = rlz.interval(units={'h', 'm', 's', 'ms', 'us', 'ns'})
    output_dtype = rlz.dtype_like('left')


@public
class TimeDiff(Binary):
    left = rlz.time
    right = rlz.time
    output_dtype = dt.Interval('s')


@public
class TimestampAdd(Binary):
    left = rlz.timestamp
    right = rlz.interval(
        units={'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'}
    )
    output_dtype = rlz.dtype_like('left')


@public
class TimestampSub(Binary):
    left = rlz.timestamp
    right = rlz.interval(
        units={'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'}
    )
    output_dtype = rlz.dtype_like('left')


@public
class TimestampDiff(Binary):
    left = rlz.timestamp
    right = rlz.timestamp
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
    left = rlz.interval
    right = rlz.interval
    op = operator.add


@public
class IntervalSubtract(IntervalBinary):
    left = rlz.interval
    right = rlz.interval
    op = operator.sub


@public
class IntervalMultiply(IntervalBinary):
    left = rlz.interval
    right = rlz.numeric
    op = operator.mul


@public
class IntervalFloorDivide(IntervalBinary):
    left = rlz.interval
    right = rlz.numeric
    op = operator.floordiv


@public
class IntervalFromInteger(Value):
    arg = rlz.integer
    unit = rlz.coerced_to(IntervalUnit)

    output_shape = rlz.shape_like("arg")

    @attribute.default
    def output_dtype(self):
        return dt.Interval(self.unit)

    @property
    def resolution(self):
        return self.output_dtype.resolution


@public
class BetweenTime(Between):
    arg = rlz.one_of([rlz.timestamp, rlz.time])
    lower_bound = rlz.one_of([rlz.time, rlz.string])
    upper_bound = rlz.one_of([rlz.time, rlz.string])


public(ExtractTimestampField=ExtractTemporalField)
