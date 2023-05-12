import datetime
import numbers
from abc import ABCMeta
from enum import Enum, EnumMeta

import dateutil.parser
import dateutil.tz
from public import public

from ibis.common.dispatch import lazy_singledispatch
from ibis.common.validators import Coercible


class ABCEnumMeta(EnumMeta, ABCMeta):
    pass


class Unit(Coercible, Enum, metaclass=ABCEnumMeta):
    @classmethod
    def __coerce__(cls, value):
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
            raise ValueError(f"Unable to coerce {value} to {cls.__name__}")

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
            'd': 'D',
            'H': 'h',
            'HH24': 'h',
            'J': 'D',
            'MI': 'm',
            'q': 'Q',
            'SYYYY': 'Y',
            'w': 'W',
            'y': 'Y',
            'YY': 'Y',
            'YYY': 'Y',
            'YYYY': 'Y',
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
    elif isinstance(tz, dateutil.tz.tzoffset):
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


@normalize_datetime.register(datetime.date)
def _from_date(value):
    return datetime.datetime(year=value.year, month=value.month, day=value.day)


@normalize_datetime.register(datetime.datetime)
def _from_datetime(value):
    return value.replace(tzinfo=normalize_timezone(value.tzinfo))


@normalize_datetime.register('pandas.Timestamp')
def _from_pandas_timestamp(value):
    # TODO(kszucs): it would make sense to preserve nanoseconds precision by
    # keeping the pandas.Timestamp object
    return value.to_pydatetime()


@normalize_datetime.register('numpy.datetime64')
def _from_numpy_datetime64(value):
    try:
        import pandas as pd
    except ImportError:
        raise TypeError("Unable to convert np.datetime64 without pandas")
    else:
        return pd.Timestamp(value).to_pydatetime()
