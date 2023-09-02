from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Literal

from public import public

import ibis.expr.datashape as ds
import ibis.expr.operations as ops
from ibis.expr.types.core import _binop
from ibis.expr.types.generic import Column, Scalar, Value
from ibis.common.annotations import annotated
import ibis.expr.datatypes as dt
from ibis import util
from ibis.common.temporal import IntervalUnit

if TYPE_CHECKING:
    import pandas as pd

    import ibis.expr.types as ir


@public
class TemporalValue(Value):
    def strftime(self, format_str: str) -> ir.StringValue:
        """Format timestamp according to `format_str`.

        Format string may depend on the backend, but we try to conform to ANSI
        `strftime`.

        Parameters
        ----------
        format_str
            `strftime` format string

        Returns
        -------
        StringValue
            Formatted version of `arg`
        """
        return ops.Strftime(self, format_str).to_expr()


@public
class TemporalScalar(Scalar, TemporalValue):
    pass


@public
class TemporalColumn(Column, TemporalValue):
    pass


class _DateComponentMixin:
    """Temporal expressions that have a date component."""

    def epoch_seconds(self) -> ir.IntegerValue:
        """Extract UNIX epoch in seconds."""
        return ops.ExtractEpochSeconds(self).to_expr()

    def year(self) -> ir.IntegerValue:
        """Extract the year component."""
        return ops.ExtractYear(self).to_expr()

    def month(self) -> ir.IntegerValue:
        """Extract the month component."""
        return ops.ExtractMonth(self).to_expr()

    def day(self) -> ir.IntegerValue:
        """Extract the day component."""
        return ops.ExtractDay(self).to_expr()

    @property
    def day_of_week(self) -> DayOfWeek:
        """A namespace of methods for extracting day of week information.

        Returns
        -------
        DayOfWeek
            An namespace expression containing methods to use to extract
            information.
        """
        return DayOfWeek(self)

    def day_of_year(self) -> ir.IntegerValue:
        """Extract the day of the year component."""
        return ops.ExtractDayOfYear(self).to_expr()

    def quarter(self) -> ir.IntegerValue:
        """Extract the quarter component."""
        return ops.ExtractQuarter(self).to_expr()

    def week_of_year(self) -> ir.IntegerValue:
        """Extract the week of the year component."""
        return ops.ExtractWeekOfYear(self).to_expr()


class _TimeComponentMixin:
    """Temporal expressions that have a time component."""

    def time(self) -> TimeValue:
        """Return the time component of the expression.

        Returns
        -------
        TimeValue
            The time component of `self`
        """
        return ops.Time(self).to_expr()

    def hour(self) -> ir.IntegerValue:
        """Extract the hour component."""
        return ops.ExtractHour(self).to_expr()

    def minute(self) -> ir.IntegerValue:
        """Extract the minute component."""
        return ops.ExtractMinute(self).to_expr()

    def second(self) -> ir.IntegerValue:
        """Extract the second component."""
        return ops.ExtractSecond(self).to_expr()

    def microsecond(self) -> ir.IntegerValue:
        """Extract the microsecond component."""
        return ops.ExtractMicrosecond(self).to_expr()

    def millisecond(self) -> ir.IntegerValue:
        """Extract the millisecond component."""
        return ops.ExtractMillisecond(self).to_expr()

    def between(
        self,
        lower: str | datetime.time | TimeValue,
        upper: str | datetime.time | TimeValue,
        timezone: str | None = None,
    ) -> ir.BooleanValue:
        """Check if the expr falls between `lower` and `upper`, inclusive.

        Adjusts according to `timezone` if provided.

        Parameters
        ----------
        lower
            Lower bound
        upper
            Upper bound
        timezone
            Time zone

        Returns
        -------
        BooleanValue
            Whether `self` is between `lower` and `upper`, adjusting `timezone`
            as needed.
        """
        op = self.op()
        if isinstance(op, ops.Time):
            # Here we pull out the first argument to the underlying Time
            # operation which is by definition (in _timestamp_value_methods) a
            # TimestampValue. We do this so that we can potentially specialize
            # the "between time" operation for
            # timestamp_value_expr.time().between(). A similar mechanism is
            # triggered when creating expressions like
            # t.column.distinct().count(), which is turned into
            # t.column.nunique().
            arg = op.arg.to_expr()
            if timezone is not None:
                arg = arg.cast(dt.Timestamp(timezone=timezone))
            op_cls = ops.BetweenTime
        else:
            arg = self
            op_cls = ops.Between

        return op_cls(arg, lower, upper).to_expr()


@public
class TimeValue(_TimeComponentMixin, TemporalValue):
    def truncate(
        self,
        unit: Literal["h", "m", "s", "ms", "us", "ns"],
    ) -> TimeValue:
        """Truncate the expression to a time expression in units of `unit`.

        Commonly used for time series resampling.

        Parameters
        ----------
        unit
            The unit to truncate to

        Returns
        -------
        TimeValue
            `self` truncated to `unit`
        """
        return ops.TimeTruncate(self, unit).to_expr()

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> TimeValue:
        """Add an interval to a time expression."""
        return _binop(ops.TimeAdd, self, other)

    add = radd = __radd__ = __add__
    """Add an interval to a time expression.

    Parameters
    ----------
    other : datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to add to time expression

    Returns
    -------
    Value : TimeValue
    """

    @annotated
    def __sub__(self, other: ops.Value[dt.Interval | dt.Time, ds.Any]):
        """Subtract a time or an interval from a time expression."""

        if other.dtype.is_time():
            op = ops.TimeDiff
        else:
            op = ops.TimeSub  # let the operation validate

        return _binop(op, self, other)

    sub = __sub__
    """Subtract a time or an interval from a time expression.

    Parameters
    ----------
    other : TimeValue | IntervalValue
        Interval to subtract from time expression

    Returns
    -------
    Value : IntervalValue | TimeValue
    """

    @annotated
    def __rsub__(self, other: ops.Value[dt.Interval | dt.Time, ds.Any]):
        """Subtract a time or an interval from a time expression."""

        if other.dtype.is_time():
            op = ops.TimeDiff
        else:
            op = ops.TimeSub  # let the operation validate

        return _binop(op, other, self)

    rsub = __rsub__


@public
class TimeScalar(TemporalScalar, TimeValue):
    pass


@public
class TimeColumn(TemporalColumn, TimeValue):
    pass


@public
class DateValue(TemporalValue, _DateComponentMixin):
    def truncate(self, unit: Literal["Y", "Q", "M", "W", "D"]) -> DateValue:
        """Truncate date expression to units of `unit`.

        Parameters
        ----------
        unit
            Unit to truncate `arg` to

        Returns
        -------
        DateValue
            Truncated date value expression
        """
        return ops.DateTruncate(self, unit).to_expr()

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> DateValue:
        """Add an interval to a date."""
        return _binop(ops.DateAdd, self, other)

    add = radd = __radd__ = __add__
    """Add an interval to a date.

    Parameters
    ----------
    other : datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to add to DateValue

    Returns
    -------
    Value : DateValue
    """

    @annotated
    def __sub__(self, other: ops.Value[dt.Date | dt.Interval, ds.Any]):
        """Subtract a date or an interval from a date."""

        if other.dtype.is_date():
            op = ops.DateDiff
        else:
            op = ops.DateSub  # let the operation validate

        return _binop(op, self, other)

    sub = __sub__
    """Subtract a date or an interval from a date.

    Parameters
    ----------
    other : datetime.date | DateValue | datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to subtract from DateValue

    Returns
    -------
    Value : DateValue
    """

    @annotated
    def __rsub__(self, other: ops.Value[dt.Date | dt.Interval, ds.Any]):
        """Subtract a date or an interval from a date."""

        if other.dtype.is_date():
            op = ops.DateDiff
        else:
            op = ops.DateSub  # let the operation validate

        return _binop(op, other, self)

    rsub = __rsub__


@public
class DateScalar(TemporalScalar, DateValue):
    pass


@public
class DateColumn(TemporalColumn, DateValue):
    pass


@public
class TimestampValue(_DateComponentMixin, _TimeComponentMixin, TemporalValue):
    def truncate(
        self,
        unit: Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"],
    ) -> TimestampValue:
        """Truncate timestamp expression to units of `unit`.

        Parameters
        ----------
        unit
            Unit to truncate to

        Returns
        -------
        TimestampValue
            Truncated timestamp expression
        """
        return ops.TimestampTruncate(self, unit).to_expr()

    def date(self) -> DateValue:
        """Return the date component of the expression.

        Returns
        -------
        DateValue
            The date component of `self`
        """
        return ops.Date(self).to_expr()

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> TimestampValue:
        """Add an interval to a timestamp."""
        return _binop(ops.TimestampAdd, self, other)

    add = radd = __radd__ = __add__
    """Add an interval to a timestamp.

    Parameters
    ----------
    other : datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to subtract from timestamp

    Returns
    -------
    Value : TimestampValue
    """

    @annotated
    def __sub__(self, other: ops.Value[dt.Timestamp | dt.Interval, ds.Any]):
        """Subtract a timestamp or an interval from a timestamp."""

        if other.dtype.is_timestamp():
            op = ops.TimestampDiff
        else:
            op = ops.TimestampSub  # let the operation validate

        return _binop(op, self, other)

    sub = __sub__
    """Subtract a timestamp or an interval from a timestamp.

    Parameters
    ----------
    other : datetime.datetime | pd.Timestamp | TimestampValue | datetime.timedelta | pd.Timedelta | IntervalValue
        Timestamp or interval to subtract from timestamp

    Returns
    -------
    Value : IntervalValue | TimestampValue
    """

    @annotated
    def __rsub__(self, other: ops.Value[dt.Timestamp | dt.Interval, ds.Any]):
        """Subtract a timestamp or an interval from a timestamp."""

        if other.dtype.is_timestamp():
            op = ops.TimestampDiff
        else:
            op = ops.TimestampSub  # let the operation validate

        return _binop(op, other, self)

    rsub = __rsub__


@public
class TimestampScalar(TemporalScalar, TimestampValue):
    pass


@public
class TimestampColumn(TemporalColumn, TimestampValue):
    pass


@public
class IntervalValue(Value):
    def to_unit(self, target_unit: str) -> IntervalValue:
        """Convert this interval to units of `target_unit`."""
        # TODO(kszucs): should use a separate operation for unit conversion
        # which we can rewrite/simplify to integer multiplication/division
        op = self.op()
        current_unit = op.dtype.unit
        target_unit = IntervalUnit.from_string(target_unit)

        if current_unit == target_unit:
            return self
        elif isinstance(op, ops.Literal):
            value = util.convert_unit(op.value, current_unit.short, target_unit.short)
            return ops.Literal(value, dtype=dt.Interval(target_unit)).to_expr()
        else:
            value = util.convert_unit(
                self.cast(dt.int64), current_unit.short, target_unit.short
            )
            return value.to_interval(target_unit)

    @property
    def years(self) -> ir.IntegerValue:
        """Extract the number of years from an interval."""
        return self.to_unit("Y")

    @property
    def quarters(self) -> ir.IntegerValue:
        """Extract the number of quarters from an interval."""
        return self.to_unit("Q")

    @property
    def months(self) -> ir.IntegerValue:
        """Extract the number of months from an interval."""
        return self.to_unit("M")

    @property
    def weeks(self) -> ir.IntegerValue:
        """Extract the number of weeks from an interval."""
        return self.to_unit("W")

    @property
    def days(self) -> ir.IntegerValue:
        """Extract the number of days from an interval."""
        return self.to_unit("D")

    @property
    def hours(self) -> ir.IntegerValue:
        """Extract the number of hours from an interval."""
        return self.to_unit("h")

    @property
    def minutes(self) -> ir.IntegerValue:
        """Extract the number of minutes from an interval."""
        return self.to_unit("m")

    @property
    def seconds(self) -> ir.IntegerValue:
        """Extract the number of seconds from an interval."""
        return self.to_unit("s")

    @property
    def milliseconds(self) -> ir.IntegerValue:
        """Extract the number of milliseconds from an interval."""
        return self.to_unit("ms")

    @property
    def microseconds(self) -> ir.IntegerValue:
        """Extract the number of microseconds from an interval."""
        return self.to_unit("us")

    @property
    def nanoseconds(self) -> ir.IntegerValue:
        """Extract the number of nanoseconds from an interval."""
        return self.to_unit("ns")

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> IntervalValue:
        """Add this interval to `other`."""
        return _binop(ops.IntervalAdd, self, other)

    add = radd = __radd__ = __add__

    def __sub__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> IntervalValue:
        """Subtract `other` from this interval."""
        return _binop(ops.IntervalSubtract, self, other)

    sub = __sub__

    def __rsub__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> IntervalValue:
        """Subtract `other` from this interval."""
        return _binop(ops.IntervalSubtract, other, self)

    rsub = __rsub__

    def __mul__(
        self,
        other: int | ir.IntegerValue,
    ) -> IntervalValue:
        """Multiply this interval by `other`."""
        return _binop(ops.IntervalMultiply, self, other)

    mul = rmul = __rmul__ = __mul__

    def __floordiv__(
        self,
        other: ir.IntegerValue,
    ) -> IntervalValue:
        """Floor-divide this interval by `other`."""
        return _binop(ops.IntervalFloorDivide, self, other)

    floordiv = __floordiv__

    def negate(self) -> ir.IntervalValue:
        """Negate an interval expression.

        Returns
        -------
        IntervalValue
            A negated interval value expression
        """
        op = self.op()
        if hasattr(op, "negate"):
            result = op.negate()
        else:
            result = ops.Negate(self)

        return result.to_expr()

    __neg__ = negate

    @staticmethod
    def __negate_op__():
        return ops.Negate


@public
class IntervalScalar(Scalar, IntervalValue):
    pass


@public
class IntervalColumn(Column, IntervalValue):
    pass


@public
class DayOfWeek:
    """A namespace of methods for extracting day of week information."""

    def __init__(self, expr):
        self._expr = expr

    def index(self):
        """Get the index of the day of the week.

        ::: {.callout-note}
        ## Ibis follows the `pandas` convention for day numbering: Monday = 0 and Sunday = 6.
        :::

        Returns
        -------
        IntegerValue
            The index of the day of the week.
        """
        return ops.DayOfWeekIndex(self._expr).to_expr()

    def full_name(self):
        """Get the name of the day of the week.

        Returns
        -------
        StringValue
            The name of the day of the week
        """
        return ops.DayOfWeekName(self._expr).to_expr()
