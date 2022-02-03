from public import public

from .core import Expr
from .generic import AnyColumn, AnyScalar, AnyValue


@public
class TemporalValue(AnyValue):
    pass  # noqa: E701,E302


@public
class TemporalScalar(AnyScalar, TemporalValue):
    pass  # noqa: E701,E302


@public
class TemporalColumn(AnyColumn, TemporalValue):
    pass  # noqa: E701,E302


@public
class TimeValue(TemporalValue):
    pass  # noqa: E701,E302


@public
class TimeScalar(TemporalScalar, TimeValue):
    pass  # noqa: E701,E302


@public
class TimeColumn(TemporalColumn, TimeValue):
    pass  # noqa: E701,E302


@public
class DateValue(TemporalValue):
    pass  # noqa: E701,E302


@public
class DateScalar(TemporalScalar, DateValue):
    pass  # noqa: E701,E302


@public
class DateColumn(TemporalColumn, DateValue):
    pass  # noqa: E701,E302


@public
class TimestampValue(TemporalValue):
    pass  # noqa: E701,E302


@public
class TimestampScalar(TemporalScalar, TimestampValue):
    pass  # noqa: E701,E302


@public
class TimestampColumn(TemporalColumn, TimestampValue):
    pass  # noqa: E701,E302


@public
class IntervalValue(AnyValue):
    pass  # noqa: E701,E302


@public
class IntervalScalar(AnyScalar, IntervalValue):
    pass  # noqa: E701,E302


@public
class IntervalColumn(AnyColumn, IntervalValue):
    pass  # noqa: E701,E302


@public
class DayOfWeek(Expr):
    def index(self):
        """Get the index of the day of the week.

        Returns
        -------
        IntegerValue
            The index of the day of the week. Ibis follows pandas conventions,
            where **Monday = 0 and Sunday = 6**.
        """
        import ibis.expr.operations as ops

        return ops.DayOfWeekIndex(self.op().arg).to_expr()

    def full_name(self):
        """Get the name of the day of the week.

        Returns
        -------
        StringValue
            The name of the day of the week
        """
        import ibis.expr.operations as ops

        return ops.DayOfWeekName(self.op().arg).to_expr()
