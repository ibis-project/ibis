"""Temporal operations."""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from public import public

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.annotations import attribute
from ibis.common.patterns import As, Attrs
from ibis.common.temporal import DateUnit, IntervalUnit, TimestampUnit, TimeUnit
from ibis.expr.operations.core import Binary, Scalar, Unary, Value
from ibis.expr.operations.logical import Between


@public
class TimestampTruncate(Value):
    """Truncate a timestamp to a specified unit."""

    arg: Value[dt.Timestamp]
    unit: IntervalUnit

    shape = rlz.shape_like("arg")
    dtype = dt.timestamp


@public
class DateTruncate(Value):
    """Truncate a date to a specified unit."""

    arg: Value[dt.Date]
    unit: DateUnit

    shape = rlz.shape_like("arg")
    dtype = dt.date


@public
class TimeTruncate(Value):
    """Truncate a time to a specified unit."""

    arg: Value[dt.Time]
    unit: TimeUnit

    shape = rlz.shape_like("arg")
    dtype = dt.time


@public
class TimestampBucket(Value):
    """Bucketize a timestamp to a specified interval."""

    arg: Value[dt.Timestamp]
    interval: Scalar[dt.Interval]
    offset: Optional[Scalar[dt.Interval]] = None

    shape = rlz.shape_like("arg")
    dtype = dt.timestamp


@public
class Strftime(Value):
    """Format a temporal value as a string."""

    arg: Value[dt.Temporal]
    format_str: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.string


@public
class StringToTimestamp(Value):
    """Convert a string to a timestamp."""

    arg: Value[dt.String]
    format_str: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.Timestamp(timezone="UTC")


@public
class StringToDate(Value):
    """Convert a string to a date."""

    arg: Value[dt.String]
    format_str: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.date


@public
class StringToTime(Value):
    """Convert a string to a time."""

    arg: Value[dt.String]
    format_str: Value[dt.String]

    shape = rlz.shape_like("args")
    dtype = dt.time


@public
class ExtractTemporalField(Unary):
    """Extract a field from a temporal value."""

    arg: Value[dt.Temporal]
    dtype = dt.int32


@public
class ExtractDateField(ExtractTemporalField):
    """Extract a field from a date."""

    arg: Value[dt.Date | dt.Timestamp]


@public
class ExtractTimeField(ExtractTemporalField):
    """Extract a field from a time."""

    arg: Value[dt.Time | dt.Timestamp]


@public
class ExtractYear(ExtractDateField):
    """Extract the year from a date or timestamp."""


@public
class ExtractIsoYear(ExtractDateField):
    """Extract the ISO year from a date or timestamp."""


@public
class ExtractMonth(ExtractDateField):
    """Extract the month from a date or timestamp."""


@public
class ExtractDay(ExtractDateField):
    """Extract the day from a date or timestamp."""


@public
class ExtractDayOfYear(ExtractDateField):
    """Extract the day of the year from a date or timestamp."""


@public
class ExtractQuarter(ExtractDateField):
    """Extract the quarter from a date or timestamp."""


@public
class ExtractEpochSeconds(ExtractDateField):
    """Extract seconds since the UNIX epoch from a date or timestamp."""


@public
class ExtractWeekOfYear(ExtractDateField):
    """Extract the week of the year from a date or timestamp."""


@public
class ExtractHour(ExtractTimeField):
    """Extract the hour from a time or timestamp."""


@public
class ExtractMinute(ExtractTimeField):
    """Extract the minute from a time or timestamp."""


@public
class ExtractSecond(ExtractTimeField):
    """Extract the second from a time or timestamp."""


@public
class ExtractMillisecond(ExtractTimeField):
    """Extract milliseconds from a time or timestamp."""


@public
class ExtractMicrosecond(ExtractTimeField):
    """Extract microseconds from a time or timestamp."""


@public
class DayOfWeekIndex(Unary):
    """Extract the index of the day of the week from a date or timestamp."""

    arg: Value[dt.Date | dt.Timestamp]

    dtype = dt.int16


@public
class DayOfWeekName(Unary):
    """Extract the name of the day of the week from a date or timestamp."""

    arg: Value[dt.Date | dt.Timestamp]

    dtype = dt.string


@public
class Time(Unary):
    """Extract the time from a timestamp."""

    dtype = dt.time


@public
class Date(Unary):
    """Extract the date from a timestamp."""

    dtype = dt.date


@public
class DateFromYMD(Value):
    """Construct a date from year, month, and day."""

    year: Value[dt.Integer]
    month: Value[dt.Integer]
    day: Value[dt.Integer]

    dtype = dt.date
    shape = rlz.shape_like("args")


@public
class TimeFromHMS(Value):
    """Construct a time from hours, minutes, and seconds."""

    hours: Value[dt.Integer]
    minutes: Value[dt.Integer]
    seconds: Value[dt.Integer]

    dtype = dt.time
    shape = rlz.shape_like("args")


@public
class TimestampFromYMDHMS(Value):
    """Construct a timestamp from components."""

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
    """Construct a timestamp from a UNIX timestamp."""

    arg: Value
    unit: TimestampUnit

    dtype = dt.timestamp
    shape = rlz.shape_like("arg")


TimeInterval = Annotated[dt.Interval, Attrs(unit=As(TimeUnit))]
DateInterval = Annotated[dt.Interval, Attrs(unit=As(DateUnit))]


@public
class DateAdd(Binary):
    """Add an interval to a date."""

    left: Value[dt.Date]
    right: Value[DateInterval]

    dtype = rlz.dtype_like("left")


@public
class DateSub(Binary):
    """Subtract an interval from a date."""

    left: Value[dt.Date]
    right: Value[DateInterval]

    dtype = rlz.dtype_like("left")


@public
class DateDiff(Binary):
    """Compute the difference between two dates."""

    left: Value[dt.Date]
    right: Value[dt.Date]

    dtype = dt.Interval("D")


@public
class TimeAdd(Binary):
    """Add an interval to a time."""

    left: Value[dt.Time]
    right: Value[TimeInterval]

    dtype = rlz.dtype_like("left")


@public
class TimeSub(Binary):
    """Subtract an interval from a time."""

    left: Value[dt.Time]
    right: Value[TimeInterval]

    dtype = rlz.dtype_like("left")


@public
class TimeDiff(Binary):
    """Compute the difference between two times."""

    left: Value[dt.Time]
    right: Value[dt.Time]

    dtype = dt.Interval("s")


@public
class TimestampAdd(Binary):
    """Add an interval to a timestamp."""

    left: Value[dt.Timestamp]
    right: Value[dt.Interval]

    dtype = rlz.dtype_like("left")


@public
class TimestampSub(Binary):
    """Subtract an interval from a timestamp."""

    left: Value[dt.Timestamp]
    right: Value[dt.Interval]

    dtype = rlz.dtype_like("left")


@public
class TimestampDiff(Binary):
    """Compute the difference between two timestamps."""

    left: Value[dt.Timestamp]
    right: Value[dt.Timestamp]

    dtype = dt.Interval("s")


@public
class IntervalBinary(Binary):
    """Base class for interval binary operations."""

    @attribute
    def dtype(self):
        interval_unit_args = [
            arg.dtype.unit for arg in (self.left, self.right) if arg.dtype.is_interval()
        ]
        unit = rlz._promote_interval_resolution(interval_unit_args)

        return self.left.dtype.copy(unit=unit)


@public
class IntervalAdd(IntervalBinary):
    """Add two intervals."""

    left: Value[dt.Interval]
    right: Value[dt.Interval]
    op = operator.add


@public
class IntervalSubtract(IntervalBinary):
    """Subtract one interval from another."""

    left: Value[dt.Interval]
    right: Value[dt.Interval]
    op = operator.sub


@public
class IntervalMultiply(IntervalBinary):
    """Multiply an interval by a scalar."""

    left: Value[dt.Interval]
    right: Value[dt.Numeric | dt.Boolean]
    op = operator.mul


@public
class IntervalFloorDivide(IntervalBinary):
    """Divide an interval by a scalar, rounding down."""

    left: Value[dt.Interval]
    right: Value[dt.Numeric | dt.Boolean]
    op = operator.floordiv


@public
class IntervalFromInteger(Value):
    """Construct an interval from an integer."""

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
    """Check if a time is between two bounds."""

    arg: Value[dt.Time | dt.Timestamp]
    lower_bound: Value[dt.Time | dt.String]
    upper_bound: Value[dt.Time | dt.String]


class TemporalDelta(Value):
    """Base class for temporal delta operations."""

    part: Value[dt.String]
    shape = rlz.shape_like("args")
    dtype = dt.int64


@public
class TimeDelta(TemporalDelta):
    """Compute the difference between two times as integer number of requested units."""

    left: Value[dt.Time]
    right: Value[dt.Time]


@public
class DateDelta(TemporalDelta):
    """Compute the difference between two dates as integer number of requested units."""

    left: Value[dt.Date]
    right: Value[dt.Date]


@public
class TimestampDelta(TemporalDelta):
    """Compute the difference between two timestamps as integer number of requested units."""

    left: Value[dt.Timestamp]
    right: Value[dt.Timestamp]


public(ExtractTimestampField=ExtractTemporalField)
