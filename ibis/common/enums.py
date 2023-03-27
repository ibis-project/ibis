from abc import ABCMeta
from enum import Enum, EnumMeta

from public import public

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
