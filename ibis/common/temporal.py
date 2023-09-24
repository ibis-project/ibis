from __future__ import annotations

import datetime
import numbers
from decimal import Decimal
from enum import Enum, EnumMeta

import dateutil.parser
import dateutil.tz
import pytz
from public import public

from ibis import util
from ibis.common.bases import AbstractMeta
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.patterns import Coercible, CoercionError


class AbstractEnumMeta(EnumMeta, AbstractMeta):
    pass


class Unit(Coercible, Enum, metaclass=AbstractEnumMeta):
    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, cls):
            return value
        else:
            return cls.from_string(value)

    @classmethod
    def from_string(cls, value):
        # TODO(kszucs): perhaps this is not needed anymore
        if isinstance(value, Unit):
            value = value.value
        elif not isinstance(value, str):
            raise CoercionError(f"Unable to coerce {value} to {cls.__name__}")

        # first look for aliases
        value = cls.aliases().get(value, value)

        # then look for the enum value (unit value)
        try:
            return cls(value)
        except ValueError:
            pass

        # then look for the enum name (unit name)
        if value.endswith("s"):
            value = value[:-1]
        try:
            return cls[value.upper()]
        except KeyError:
            raise CoercionError(f"Unable to coerce {value} to {cls.__name__}")

    @classmethod
    def aliases(cls):
        return {}

    @property
    def singular(self) -> str:
        return self.name.lower()

    @property
    def plural(self) -> str:
        return self.singular + "s"

    @property
    def short(self) -> str:
        return self.value


class TemporalUnit(Unit):
    @classmethod
    def aliases(cls):
        return {
            "d": "D",
            "H": "h",
            "HH24": "h",
            "J": "D",
            "MI": "m",
            "q": "Q",
            "SYYYY": "Y",
            "w": "W",
            "y": "Y",
            "YY": "Y",
            "YYY": "Y",
            "YYYY": "Y",
        }


@public
class DateUnit(TemporalUnit):
    YEAR = "Y"
    QUARTER = "Q"
    MONTH = "M"
    WEEK = "W"
    DAY = "D"


@public
class TimeUnit(TemporalUnit):
    HOUR = "h"
    MINUTE = "m"
    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"


@public
class TimestampUnit(TemporalUnit):
    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"


@public
class IntervalUnit(TemporalUnit):
    YEAR = "Y"
    QUARTER = "Q"
    MONTH = "M"
    WEEK = "W"
    DAY = "D"
    HOUR = "h"
    MINUTE = "m"
    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"

    def is_date(self) -> bool:
        return self.name in DateUnit.__members__

    def is_time(self) -> bool:
        return self.name in TimeUnit.__members__


def normalize_timedelta(
    value: datetime.timedelta | numbers.Real, unit: IntervalUnit
) -> datetime.timedelta:
    """Normalize a timedelta value to the given unit.

    Parameters
    ----------
    value
        The value to normalize, either a timedelta or a number.
    unit
        The unit to normalize to.

    Returns
    -------
    The normalized timedelta value.

    Examples
    --------
    >>> from datetime import timedelta
    >>> normalize_timedelta(1, IntervalUnit.SECOND)
    1
    >>> normalize_timedelta(1, IntervalUnit.DAY)
    1
    >>> normalize_timedelta(timedelta(days=14), IntervalUnit.WEEK)
    2
    >>> normalize_timedelta(timedelta(seconds=3), IntervalUnit.MILLISECOND)
    3000
    >>> normalize_timedelta(timedelta(seconds=3), IntervalUnit.MICROSECOND)
    3000000
    """
    if isinstance(value, datetime.timedelta):
        # datetime.timedelta only stores days, seconds, and microseconds internally
        if value.days and not (value.seconds or value.microseconds):
            value = util.convert_unit(value.days, "D", unit.short, floor=False)
        else:
            total_seconds = Decimal(str(value.total_seconds()))
            value = util.convert_unit(total_seconds, "s", unit.short, floor=False)
    else:
        value = Decimal(value)

    # check that value is integral
    if value % 1 != 0:
        raise ValueError(f"Normalizing {value} to {unit} would lose precision")

    return int(value)


def normalize_timezone(tz):
    if tz is None:
        return None
    elif isinstance(tz, str):
        if tz == "UTC":
            return dateutil.tz.tzutc()
        else:
            return dateutil.tz.gettz(tz)
    elif isinstance(tz, (int, float)):
        return datetime.timezone(datetime.timedelta(hours=tz))
    elif isinstance(tz, (dateutil.tz.tzoffset, pytz._FixedOffset)):
        # this way we have a proper tzname() output, e.g. "UTC+01:00"
        return datetime.timezone(tz.utcoffset(None))
    elif isinstance(tz, datetime.tzinfo):
        return tz
    else:
        raise TypeError(f"Unable to normalize {type(tz)} to timezone")


@lazy_singledispatch
def normalize_datetime(value):
    raise TypeError(f"Unable to normalize {type(value)} to timestamp")


@normalize_datetime.register(str)
def _from_str(value):
    lower = value.lower()
    if lower == "now":
        return datetime.datetime.now()
    elif lower == "today":
        return datetime.datetime.today()

    value = dateutil.parser.parse(value)
    return value.replace(tzinfo=normalize_timezone(value.tzinfo))


@normalize_datetime.register(numbers.Number)
def _from_number(value):
    return datetime.datetime.utcfromtimestamp(value)


@normalize_datetime.register(datetime.time)
def _from_time(value):
    return datetime.datetime.combine(datetime.date.today(), value)


@normalize_datetime.register(datetime.date)
def _from_date(value):
    return datetime.datetime(year=value.year, month=value.month, day=value.day)


@normalize_datetime.register(datetime.datetime)
def _from_datetime(value):
    return value.replace(tzinfo=normalize_timezone(value.tzinfo))


@normalize_datetime.register("pandas.Timestamp")
def _from_pandas_timestamp(value):
    # TODO(kszucs): it would make sense to preserve nanoseconds precision by
    # keeping the pandas.Timestamp object
    return value.to_pydatetime()


@normalize_datetime.register("numpy.datetime64")
def _from_numpy_datetime64(value):
    try:
        import pandas as pd
    except ImportError:
        raise TypeError("Unable to convert np.datetime64 without pandas")
    else:
        return pd.Timestamp(value).to_pydatetime()
