from __future__ import annotations

import operator

import toolz
from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis import util
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Binary, Unary, Value
from ibis.expr.operations.generic import Cast
from ibis.expr.operations.logical import Between


@public
class TemporalUnary(Unary):
    arg = rlz.temporal


@public
class TimestampUnary(Unary):
    arg = rlz.timestamp


_date_units = {
    'Y': 'Y',
    'y': 'Y',
    'year': 'Y',
    'YEAR': 'Y',
    'YYYY': 'Y',
    'SYYYY': 'Y',
    'YYY': 'Y',
    'YY': 'Y',
    'Q': 'Q',
    'q': 'Q',
    'quarter': 'Q',
    'QUARTER': 'Q',
    'M': 'M',
    'month': 'M',
    'MONTH': 'M',
    'w': 'W',
    'W': 'W',
    'week': 'W',
    'WEEK': 'W',
    'd': 'D',
    'D': 'D',
    'J': 'D',
    'day': 'D',
    'DAY': 'D',
}

_time_units = {
    'h': 'h',
    'H': 'h',
    'HH24': 'h',
    'hour': 'h',
    'HOUR': 'h',
    'm': 'm',
    'MI': 'm',
    'minute': 'm',
    'MINUTE': 'm',
    's': 's',
    'second': 's',
    'SECOND': 's',
    'ms': 'ms',
    'millisecond': 'ms',
    'MILLISECOND': 'ms',
    'us': 'us',
    'microsecond': 'ms',
    'MICROSECOND': 'ms',
    'ns': 'ns',
    'nanosecond': 'ns',
    'NANOSECOND': 'ns',
}

_timestamp_units = toolz.merge(_date_units, _time_units)


@public
class TimestampTruncate(Value):
    arg = rlz.timestamp
    unit = rlz.map_to(_timestamp_units)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.timestamp


@public
class DateTruncate(Value):
    arg = rlz.date
    unit = rlz.map_to(_date_units)

    output_shape = rlz.shape_like("arg")
    output_dtype = dt.date


@public
class TimeTruncate(Value):
    arg = rlz.time
    unit = rlz.map_to(_time_units)

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
    unit = rlz.isin({'s', 'ms', 'us', 'ns'})

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
class ToIntervalUnit(Value):
    arg = rlz.interval
    unit = rlz.isin({'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'})

    output_shape = rlz.shape_like("arg")

    def __init__(self, arg, unit):
        dtype = arg.output_dtype

        # TODO(kszucs): remove the expression wrapping required for arithmetic
        # overloads
        if dtype.unit != unit:
            arg = util.convert_unit(arg, dtype.unit, unit)
        super().__init__(arg=arg, unit=unit)

    @attribute.default
    def output_dtype(self):
        return self.arg.output_dtype.copy(unit=self.unit)


@public
class IntervalBinary(Binary):
    @attribute.default
    def output_dtype(self):
        integer_args = [
            Cast(arg, to=arg.output_dtype.value_type)
            if arg.output_dtype.is_interval()
            else arg
            for arg in (self.left, self.right)
        ]
        value_dtype = rlz._promote_integral_binop(integer_args, self.op)

        return self.left.output_dtype.copy(value_type=value_dtype)


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
    unit = rlz.isin({'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'})

    output_shape = rlz.shape_like("arg")

    @attribute.default
    def output_dtype(self):
        return dt.Interval(self.unit, value_type=self.arg.output_dtype)

    @property
    def resolution(self):
        return self.output_dtype.resolution


@public
class BetweenTime(Between):
    arg = rlz.one_of([rlz.timestamp, rlz.time])
    lower_bound = rlz.one_of([rlz.time, rlz.string])
    upper_bound = rlz.one_of([rlz.time, rlz.string])


public(ExtractTimestampField=ExtractTemporalField)
