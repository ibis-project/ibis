from __future__ import annotations

import datetime
import math
from abc import ABC, abstractmethod
from collections import defaultdict

from ibis.common.temporal import IntervalUnit
from ibis.util import convert_unit

# For details on what precisions Flink SQL interval types support, see
# https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/dev/table/types/#interval-year-to-month
# https://nightlies.apache.org/flink/flink-docs-release-1.17/docs/dev/table/types/#interval-day-to-second
MIN_ALLOWED_PRECISION = {
    IntervalUnit.YEAR: 1,
    IntervalUnit.MONTH: 2,
    IntervalUnit.DAY: 1,
    IntervalUnit.HOUR: 2,
    IntervalUnit.MINUTE: 2,
    IntervalUnit.SECOND: 0,
}

MAX_ALLOWED_PRECISION = {
    IntervalUnit.YEAR: 4,
    IntervalUnit.MONTH: 2,
    IntervalUnit.DAY: 6,
    IntervalUnit.HOUR: 2,
    IntervalUnit.MINUTE: 2,
    IntervalUnit.SECOND: 9,
}


MICROSECONDS_IN_UNIT = {
    unit: datetime.timedelta(**{unit.plural: 1}).total_seconds() * 10**6
    for unit in [
        IntervalUnit.DAY,
        IntervalUnit.HOUR,
        IntervalUnit.MINUTE,
        IntervalUnit.SECOND,
    ]
}
MICROSECONDS_IN_UNIT[IntervalUnit.MONTH] = (
    30 * datetime.timedelta(days=1).total_seconds() * 10**6
)
MICROSECONDS_IN_UNIT[IntervalUnit.YEAR] = (
    365 * datetime.timedelta(days=1).total_seconds() * 10**6
)


def _calculate_precision(interval_value: int) -> int:
    """Calculate interval precision.

    FlinkSQL interval data types use leading precision and fractional-
    seconds precision. Because the leading precision defaults to 2, we need to
    specify a different precision when the value exceeds 2 digits.

    (see
    https://learn.microsoft.com/en-us/sql/odbc/reference/appendixes/interval-literals)
    """
    # log10(interval_value) + 1 is equivalent to len(str(interval_value)), but is significantly
    # faster and more memory-efficient
    if interval_value == 0:
        return 0
    if interval_value < 0:
        raise ValueError(
            f"Expecting value to be a non-negative integer, got {interval_value}"
        )
    return int(math.log10(interval_value)) + 1


def _format_value_with_precision(value: int, precision: int) -> str:
    """Format value so that it fills a specified precision."""
    return str(value).zfill(precision)


def format_precision(precision: int, unit: IntervalUnit) -> str:
    """Format precision values in Flink SQL."""
    if precision > MAX_ALLOWED_PRECISION[unit]:
        raise ValueError(
            f"{precision} is bigger than the allowed precision for {unit} ({MAX_ALLOWED_PRECISION[unit]})"
        )

    return '' if precision <= 2 else f'({precision})'


class FlinkIntervalType(ABC):
    """Abstract Base Class for Flink interval type.

    Flink supports only two types of temporal intervals: day-time intervals with up to nanosecond
    granularity or year-month intervals with up to month granularity.

    This Abstract Base Class provides functionality so that a given IntervalType instance can be
    translated appropriately into Flink SQL.
    """

    def __init__(self, value: int, unit: str) -> None:
        self.value = value
        self.unit = unit
        self.interval_segments = self._convert_to_combined_units()
        self.precisions = self._calculate_precisions()

    @classmethod
    @property
    @abstractmethod
    def units(self):
        ...

    @classmethod
    @property
    @abstractmethod
    def factors(self):
        ...

    @abstractmethod
    def _convert_to_highest_resolution(self):
        ...

    def _convert_to_combined_units(self) -> dict:
        converted_total = self._convert_to_highest_resolution()
        interval_segments = defaultdict(int)

        rem = converted_total
        for unit, factor in zip(self.units[:-1], self.factors):
            q, rem = divmod(rem, factor)
            interval_segments[unit] = int(q)
            if rem == 0:
                break
        if rem > 0:
            interval_segments[self.units[-1]] = rem
        return interval_segments

    @abstractmethod
    def _calculate_precisions(self) -> dict:
        ...

    @abstractmethod
    def format_as_string(self, interval_segments: dict, precisions: dict) -> str:
        ...


class YearsToMonthsInterval(FlinkIntervalType):
    units = (IntervalUnit.YEAR, IntervalUnit.MONTH)
    factors = (12,)

    def _convert_to_highest_resolution(self) -> int:
        return convert_unit(self.value, self.unit, to=IntervalUnit.MONTH.value)

    def _calculate_precisions(self) -> dict:
        precisions = {}
        for unit in self.units:
            value = self.interval_segments[unit]
            prec = _calculate_precision(value)
            precisions[unit] = max(prec, 2)
        return precisions

    def format_as_string(self) -> str:
        years = self.interval_segments[IntervalUnit.YEAR]
        months = self.interval_segments[IntervalUnit.MONTH]
        return (
            f"'{_format_value_with_precision(years, self.precisions[IntervalUnit.YEAR])}"
            f"-{_format_value_with_precision(months, self.precisions[IntervalUnit.MONTH])}' YEAR"
            f"{format_precision(self.precisions[IntervalUnit.YEAR], IntervalUnit.YEAR)} "
            "TO MONTH"
            f"{format_precision(self.precisions[IntervalUnit.MONTH], IntervalUnit.MONTH)}"
        )


class DaysToSecondsInterval(FlinkIntervalType):
    units = (
        IntervalUnit.DAY,
        IntervalUnit.HOUR,
        IntervalUnit.MINUTE,
        IntervalUnit.SECOND,
        IntervalUnit.MICROSECOND,
    )
    factors = (86400 * 10**6, 3600 * 10**6, 60 * 10**6, 10**6)

    def _convert_to_highest_resolution(self) -> int:
        return convert_unit(self.value, self.unit, to=IntervalUnit.MICROSECOND.value)

    def _calculate_precisions(self) -> dict:
        precisions = {}
        for unit in self.units:
            value = self.interval_segments[unit]
            if unit != IntervalUnit.MICROSECOND:
                prec = _calculate_precision(value)
                precisions[unit] = max(prec, 2)
            else:
                precisions[IntervalUnit.MICROSECOND] = max(prec, 6)
        return precisions

    def format_as_string(self) -> str:
        days = self.interval_segments[IntervalUnit.DAY]
        hours = self.interval_segments[IntervalUnit.HOUR]
        minutes = self.interval_segments[IntervalUnit.MINUTE]
        seconds = self.interval_segments[IntervalUnit.SECOND]
        microseconds = self.interval_segments[IntervalUnit.MICROSECOND]

        return (
            f"'{_format_value_with_precision(days, self.precisions[IntervalUnit.DAY])} "
            f"{_format_value_with_precision(hours, self.precisions[IntervalUnit.HOUR])}:"
            f"{_format_value_with_precision(minutes, self.precisions[IntervalUnit.MINUTE])}:"
            f"{_format_value_with_precision(seconds, self.precisions[IntervalUnit.SECOND])}."
            f"{_format_value_with_precision(microseconds, self.precisions[IntervalUnit.MICROSECOND])}' "
            f"DAY{format_precision(self.precisions[IntervalUnit.DAY], IntervalUnit.DAY)} "
            "TO SECOND"
            f"{format_precision(self.precisions[IntervalUnit.SECOND], IntervalUnit.SECOND)}"
        )
