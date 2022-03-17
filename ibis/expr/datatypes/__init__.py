from __future__ import annotations

import abc
import ast
import builtins
import collections
import datetime
import decimal
import enum
import functools
import numbers
import re
import typing
import uuid as _uuid
from typing import (
    AbstractSet,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    Sequence,
    TypeVar,
)

import numpy as np
import pandas as pd
import parsy as p
import toolz
from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.types as ir
from ibis import util

try:
    import shapely.geometry

    IS_SHAPELY_AVAILABLE = True
except ImportError:
    IS_SHAPELY_AVAILABLE = False


class DataType(util.CachedEqMixin):
    """Base class for all data types."""

    nullable: bool
    """Whether the data type can hold `NULL` values."""
    _pretty: str
    """Pretty representation of the datatype."""
    _hash: int | None
    """Hash of the datatype."""

    __slots__ = "nullable", "_pretty", "_hash"
    _ignored_hash_slots = "_pretty", "_hash"

    def __init__(self, nullable: bool = True, **kwargs) -> None:
        self.nullable = nullable
        self._pretty = self.name.lower()
        self._hash = None

    def __call__(self, nullable: bool = True) -> DataType:
        if nullable is not True and nullable is not False:
            raise TypeError(
                "__call__ only accepts the 'nullable' argument. "
                "Please construct a new instance of the type to change the "
                "values of the attributes."
            )
        return self._factory(nullable=nullable)

    @property
    def _slot_list(self) -> tuple[str, ...]:
        return tuple(
            slot
            for slot in toolz.unique((*self.__slots__, "nullable"))
            if slot not in self._ignored_hash_slots
        )

    def _factory(self, nullable: bool = True) -> DataType:
        slots = {
            slot: getattr(self, slot)
            for slot in self._slot_list
            if slot != "nullable"
        }
        return self.__class__(nullable=nullable, **slots)

    def equals(
        self,
        other: typing.Any,
        cache: MutableMapping[Hashable, bool] | None = None,
    ) -> bool:
        if not isinstance(other, DataType):
            raise TypeError(
                'Comparing datatypes to other types is not allowed. Convert '
                f'{other!r} to the equivalent DataType instance.'
            )
        return super().equals(other, cache=cache)

    def _make_hash(self) -> int:
        custom_parts = tuple(getattr(self, slot) for slot in self._slot_list)
        return hash((self.__class__, *custom_parts, self.nullable))

    def __hash__(self) -> int:
        if (result := self._hash) is None:
            self._hash = result = self._make_hash()
        return result

    def __repr__(self) -> str:
        args = ", ".join(
            f"{slot}={getattr(self, slot)!r}" for slot in self._slot_list
        )
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        prefix = "!" * (not self.nullable)
        return f"{prefix}{self._pretty}"

    @property
    def name(self) -> str:
        """Return the name of the data type."""
        return self.__class__.__name__

    def __component_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.nullable == other.nullable and self.__fields_eq__(
            other, cache=cache
        )

    @abc.abstractmethod
    def __fields_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        """Return whether the fields of two datatypes are equal."""

    def _type_check(self, other: typing.Any) -> None:
        if not isinstance(other, DataType):
            raise TypeError(
                "invalid equality comparison between "
                f"DataType and {type(other)}"
            )

    def castable(self, target, **kwargs):
        """Return whether this data type is castable to `target`."""
        return castable(self, target, **kwargs)

    def cast(self, target, **kwargs):
        """Cast this data type to `target`."""
        return cast(self, target, **kwargs)

    def scalar_type(self):
        """Return a scalar expression with this data type."""
        return functools.partial(self.scalar, dtype=self)

    def column_type(self):
        """Return a column expression with this data type."""
        return functools.partial(self.column, dtype=self)


class Any(DataType):
    """Values of any type."""

    __slots__ = ()

    def __fields_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return True


class Primitive(DataType):
    """Values with known size."""

    __slots__ = ()

    def __fields_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return True


class Null(DataType):
    """Null values."""

    scalar = ir.NullScalar
    column = ir.NullColumn

    __slots__ = ()

    def __fields_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return True


class Variadic(DataType):
    """Values with unknown size."""

    __slots__ = ()

    def __fields_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return True


class Boolean(Primitive):
    """True or False values."""

    scalar = ir.BooleanScalar
    column = ir.BooleanColumn

    __slots__ = ()


class Bounds(NamedTuple):
    lower: int
    upper: int


class Integer(Primitive):
    """Integer values."""

    scalar = ir.IntegerScalar
    column = ir.IntegerColumn

    __slots__ = ()

    @property
    def _nbytes(self) -> int:
        """Return the number of bytes used to store values of this type."""
        raise TypeError(
            "Cannot determine the size in bytes of an abstract integer type."
        )


class String(Variadic):
    """A type representing a string.

    Notes
    -----
    Because of differences in the way different backends handle strings, we
    cannot assume that strings are UTF-8 encoded.
    """

    scalar = ir.StringScalar
    column = ir.StringColumn

    __slots__ = ()


class Binary(Variadic):
    """A type representing a sequence of bytes.

    Notes
    -----
    Some databases treat strings and blobs of equally, and some do not.

    For example, Impala doesn't make a distinction between string and binary
    types but PostgreSQL has a `TEXT` type and a `BYTEA` type which are
    distinct types that have different behavior.
    """

    scalar = ir.BinaryScalar
    column = ir.BinaryColumn

    __slots__ = ()


class Date(Primitive):
    """Date values."""

    scalar = ir.DateScalar
    column = ir.DateColumn

    __slots__ = ()


class Time(Primitive):
    """Time values."""

    scalar = ir.TimeScalar
    column = ir.TimeColumn

    __slots__ = ()


class Timestamp(DataType):
    """Timestamp values."""

    timezone: str
    """The timezone of values of this type."""

    scalar = ir.TimestampScalar
    column = ir.TimestampColumn

    __slots__ = ('timezone',)

    def __init__(
        self, timezone: str | None = None, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.timezone = timezone
        if timezone is not None:
            self._pretty += f"({timezone!r})"

    def __fields_eq__(
        self,
        other: Timestamp,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.timezone == other.timezone


class SignedInteger(Integer):
    """Signed integer values."""

    @property
    def largest(self):
        """Return the largest type of signed integer."""
        return int64

    @property
    def bounds(self):
        exp = self._nbytes * 8 - 1
        upper = (1 << exp) - 1
        return Bounds(lower=~upper, upper=upper)


class UnsignedInteger(Integer):
    """Unsigned integer values."""

    @property
    def largest(self):
        """Return the largest type of unsigned integer."""
        return uint64

    @property
    def bounds(self):
        exp = self._nbytes * 8 - 1
        upper = 1 << exp
        return Bounds(lower=0, upper=upper)


class Floating(Primitive):
    """Floating point values."""

    scalar = ir.FloatingScalar
    column = ir.FloatingColumn

    __slots__ = ()

    @property
    def largest(self):
        """Return the largest type of floating point values."""
        return float64

    @property
    def _nbytes(self) -> int:
        raise TypeError(
            "Cannot determine the size in bytes of an abstract floating "
            "point type."
        )


class Int8(SignedInteger):
    """Signed 8-bit integers."""

    __slots__ = ()
    _nbytes = 1


class Int16(SignedInteger):
    """Signed 16-bit integers."""

    __slots__ = ()
    _nbytes = 2


class Int32(SignedInteger):
    """Signed 32-bit integers."""

    __slots__ = ()
    _nbytes = 4


class Int64(SignedInteger):
    """Signed 64-bit integers."""

    __slots__ = ()
    _nbytes = 8


class UInt8(UnsignedInteger):
    """Unsigned 8-bit integers."""

    __slots__ = ()
    _nbytes = 1


class UInt16(UnsignedInteger):
    """Unsigned 16-bit integers."""

    __slots__ = ()
    _nbytes = 2


class UInt32(UnsignedInteger):
    """Unsigned 32-bit integers."""

    __slots__ = ()
    _nbytes = 4


class UInt64(UnsignedInteger):
    """Unsigned 64-bit integers."""

    __slots__ = ()
    _nbytes = 8


class Float16(Floating):
    """16-bit floating point numbers."""

    __slots__ = ()
    _nbytes = 2


class Float32(Floating):
    """32-bit floating point numbers."""

    __slots__ = ()
    _nbytes = 4


class Float64(Floating):
    """64-bit floating point numbers."""

    __slots__ = ()
    _nbytes = 8


Halffloat = Float16
Float = Float32
Double = Float64


class Decimal(DataType):
    """Fixed-precision decimal values."""

    precision: int
    """The number of values after the decimal point."""

    scale: int
    """The number of decimal places values of this type can hold."""

    scalar = ir.DecimalScalar
    column = ir.DecimalColumn

    __slots__ = 'precision', 'scale'

    def __init__(
        self, precision: int, scale: int, nullable: bool = True
    ) -> None:
        if not isinstance(precision, numbers.Integral):
            raise TypeError('Decimal type precision must be an integer')
        if not isinstance(scale, numbers.Integral):
            raise TypeError('Decimal type scale must be an integer')
        if precision < 0:
            raise ValueError('Decimal type precision cannot be negative')
        if not precision:
            raise ValueError('Decimal type precision cannot be zero')
        if scale < 0:
            raise ValueError('Decimal type scale cannot be negative')
        if precision < scale:
            raise ValueError(
                'Decimal type precision must be greater than or equal to '
                'scale. Got precision={:d} and scale={:d}'.format(
                    precision, scale
                )
            )

        super().__init__(nullable=nullable)
        self.precision = precision  # type: int
        self.scale = scale  # type: int
        self._pretty += f"({self.precision:d}, {self.scale:d})"

    @property
    def largest(self) -> Decimal:
        """Return the largest decimal type."""
        return Decimal(38, self.scale)

    def __fields_eq__(
        self,
        other: Decimal,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.precision == other.precision and self.scale == other.scale


class Interval(DataType):
    """Interval values."""

    value_type: DataType
    """The underlying type of the stored values."""
    unit: str
    """The time unit of the interval."""

    scalar = ir.IntervalScalar
    column = ir.IntervalColumn

    __slots__ = 'value_type', 'unit'

    # based on numpy's units
    _units = {
        'Y': 'year',
        'Q': 'quarter',
        'M': 'month',
        'W': 'week',
        'D': 'day',
        'h': 'hour',
        'm': 'minute',
        's': 'second',
        'ms': 'millisecond',
        'us': 'microsecond',
        'ns': 'nanosecond',
    }

    _timedelta_to_interval_units = {
        'days': 'D',
        'hours': 'h',
        'minutes': 'm',
        'seconds': 's',
        'milliseconds': 'ms',
        'microseconds': 'us',
        'nanoseconds': 'ns',
    }

    def _convert_timedelta_unit_to_interval_unit(self, unit: str):
        if unit not in self._timedelta_to_interval_units:
            raise ValueError
        return self._timedelta_to_interval_units[unit]

    def __init__(
        self,
        unit: str = 's',
        value_type: Integer = None,
        nullable: bool = True,
    ) -> None:
        super().__init__(nullable=nullable)
        if unit not in self._units:
            try:
                unit = self._convert_timedelta_unit_to_interval_unit(unit)
            except ValueError:
                raise ValueError(f'Unsupported interval unit `{unit}`')

        if value_type is None:
            value_type = int32
        else:
            value_type = dtype(value_type)

        if not isinstance(value_type, Integer):
            raise TypeError("Interval's inner type must be an Integer subtype")

        self.unit = unit
        self.value_type = value_type
        self._pretty += f"<{self.value_type}>(unit={self.unit!r})"

    @property
    def bounds(self):
        return self.value_type.bounds

    @property
    def resolution(self):
        """The interval unit's name."""
        return self._units[self.unit]

    def __fields_eq__(
        self,
        other: Interval,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.unit == other.unit and self.value_type.equals(
            other.value_type,
            cache=cache,
        )


class Category(DataType):
    scalar = ir.CategoryScalar
    column = ir.CategoryColumn

    __slots__ = ('cardinality',)

    def __init__(self, cardinality=None, nullable=True):
        super().__init__(nullable=nullable)
        self.cardinality = cardinality

    def __repr__(self):
        if self.cardinality is not None:
            cardinality = self.cardinality
        else:
            cardinality = 'unknown'
        return f'{self.name}(cardinality={cardinality!r})'

    def to_integer_type(self):
        # TODO: this should be removed I guess
        if self.cardinality is None:
            return int64
        else:
            return infer(self.cardinality)

    def __fields_eq__(
        self,
        other: Category,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.cardinality == other.cardinality


class Struct(DataType):
    """Structured values."""

    names: list[str]
    """Field names of the struct."""
    types: Sequence[DataType]
    """Types of the fields of the struct."""

    scalar = ir.StructScalar
    column = ir.StructColumn

    __slots__ = 'names', 'types'

    def __init__(
        self,
        names: Iterable[str],
        types: Iterable[DataType],
        nullable: bool = True,
    ) -> None:
        """Construct a struct type from `names` and `types`.

        Parameters
        ----------
        names
            Sequence of strings indicating the name of each field in the
            struct.
        types
            Sequence of strings or :class:`~ibis.expr.datatypes.DataType`
            instances, one for each field
        nullable
            Whether the struct can be null
        """
        if not (names and types):
            raise ValueError('names and types must not be empty')
        if len(names) != len(types):
            raise ValueError('names and types must have the same length')

        super().__init__(nullable=nullable)
        self.names = names
        self.types = types
        self._pretty += "<{}>".format(
            ", ".join(map("{}: {}".format, self.names, self.types))
        )

    @classmethod
    def from_tuples(
        cls,
        pairs: Iterable[tuple[str, str | DataType]],
        nullable: bool = True,
    ) -> Struct:
        names, types = zip(*pairs)
        return cls(list(names), list(map(dtype, types)), nullable=nullable)

    @classmethod
    def from_dict(
        cls,
        pairs: Mapping[str, str | DataType],
        nullable: bool = True,
    ) -> Struct:
        names, types = pairs.keys(), pairs.values()
        return cls(list(names), list(map(dtype, types)), nullable=nullable)

    @property
    def pairs(self) -> Mapping:
        return dict(zip(self.names, self.types))

    def __getitem__(self, key: str) -> DataType:
        return self.pairs[key]

    def _make_hash(self) -> int:
        return hash(
            (
                self.__class__,
                tuple(self.names),
                tuple(self.types),
                self.nullable,
            )
        )

    def __fields_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.names == other.names and util.seq_eq(
            self.types,
            other.types,
            cache=cache,
        )

    def __repr__(self) -> str:
        return '{}({}, nullable={})'.format(
            self.name, list(self.pairs.items()), self.nullable
        )


class Array(Variadic):
    """Array values."""

    value_type: DataType
    """The type of the elements of the array."""
    scalar = ir.ArrayScalar
    column = ir.ArrayColumn

    __slots__ = ('value_type',)

    def __init__(
        self, value_type: str | DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.value_type = dtype(value_type)
        self._pretty += f"<{self.value_type}>"

    def __fields_eq__(
        self,
        other: Array,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.value_type.equals(other.value_type, cache=cache)


class Set(Variadic):
    """Set values."""

    value_type: DataType
    """The type of the elements of the set."""
    scalar = ir.SetScalar
    column = ir.SetColumn

    __slots__ = ('value_type',)

    def __init__(
        self, value_type: str | DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.value_type = dtype(value_type)
        self._pretty += f"<{self.value_type}>"

    def __fields_eq__(
        self,
        other: Set,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.value_type.equals(other.value_type, cache=cache)


class Enum(DataType):
    """Enumeration values."""

    rep_type: DataType
    """The type of the key of the enumeration."""
    value_type: DataType
    """The type of the elements of the enumeration."""

    scalar = ir.EnumScalar
    column = ir.EnumColumn

    __slots__ = 'rep_type', 'value_type'

    def __init__(
        self, rep_type: DataType, value_type: DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.rep_type = dtype(rep_type)
        self.value_type = dtype(value_type)

    def __fields_eq__(
        self,
        other: Enum,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.rep_type.equals(
            other.rep_type,
            cache=cache,
        ) and self.value_type.equals(other.value_type, cache=cache)


class Map(Variadic):
    """Associative array values."""

    key_type: DataType
    """The type of the key of the map."""
    value_type: DataType
    """The type of the values of the map."""
    scalar = ir.MapScalar
    column = ir.MapColumn

    __slots__ = 'key_type', 'value_type'

    def __init__(
        self, key_type: DataType, value_type: DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.key_type = dtype(key_type)
        self.value_type = dtype(value_type)
        self._pretty += f"<{self.key_type}, {self.value_type}>"

    def __fields_eq__(
        self,
        other: Map,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.key_type.equals(
            other.key_type,
            cache=cache,
        ) and self.value_type.equals(other.value_type, cache=cache)


class JSON(String):
    """JSON values."""

    scalar = ir.JSONScalar
    column = ir.JSONColumn


class JSONB(Binary):
    """JSON data stored in a binary representation.

    This representation eliminates whitespace, duplicate keys, and does not
    preserve key ordering.
    """

    scalar = ir.JSONBScalar
    column = ir.JSONBColumn


class GeoSpatial(DataType):
    """Geospatial values."""

    __slots__ = 'geotype', 'srid'

    geotype: Literal['geography', 'geometry'] | None
    """The specific geospatial type"""
    srid: int | None
    """The spatial reference identifier."""
    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    def __init__(
        self,
        geotype: Literal['geography', 'geometry'] | None = None,
        srid: int | None = None,
        nullable: bool = True,
    ) -> None:
        """Geospatial data type base class

        Parameters
        ----------
        geotype
            Specification of geospatial type which could be `geography` or
            `geometry`.
        srid
            Spatial Reference System Identifier
        nullable
            Whether the value can be null
        """
        super().__init__(nullable=nullable)

        if geotype not in (None, 'geometry', 'geography'):
            raise ValueError(
                'The `geotype` parameter should be `geometry` or `geography`'
            )

        self.geotype = geotype
        self.srid = srid

        if self.geotype is not None:
            self._pretty += ':' + self.geotype
        if self.srid is not None:
            self._pretty += ';' + str(self.srid)

    def __fields_eq__(
        self,
        other: GeoSpatial,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return self.geotype == other.geotype and self.srid == other.srid


class Geometry(GeoSpatial):
    """Geometry values."""

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geotype = 'geometry'
        self._pretty = self.name.lower()


class Geography(GeoSpatial):
    """Geography values."""

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geotype = 'geography'
        self._pretty = self.name.lower()


class Point(GeoSpatial):
    """A point described by two coordinates."""

    scalar = ir.PointScalar
    column = ir.PointColumn

    __slots__ = ()


class LineString(GeoSpatial):
    """A sequence of 2 or more points."""

    scalar = ir.LineStringScalar
    column = ir.LineStringColumn

    __slots__ = ()


class Polygon(GeoSpatial):
    """A set of one or more closed line strings.

    The first line string represents the shape (external ring) and the rest
    represent holes in that shape (internal rings).
    """

    scalar = ir.PolygonScalar
    column = ir.PolygonColumn

    __slots__ = ()


class MultiLineString(GeoSpatial):
    """A set of one or more line strings."""

    scalar = ir.MultiLineStringScalar
    column = ir.MultiLineStringColumn

    __slots__ = ()


class MultiPoint(GeoSpatial):
    """A set of one or more points."""

    scalar = ir.MultiPointScalar
    column = ir.MultiPointColumn

    __slots__ = ()


class MultiPolygon(GeoSpatial):
    """A set of one or more polygons."""

    scalar = ir.MultiPolygonScalar
    column = ir.MultiPolygonColumn

    __slots__ = ()


class UUID(DataType):
    """A 128-bit number used to identify information in computer systems."""

    scalar = ir.UUIDScalar
    column = ir.UUIDColumn

    __slots__ = ()

    def __fields_eq__(
        self,
        other: DataType,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return True


class MACADDR(String):
    """Media Access Control (MAC) address of a network interface."""

    scalar = ir.MACADDRScalar
    column = ir.MACADDRColumn

    __slots__ = ()


class INET(String):
    """IP addresses."""

    scalar = ir.INETScalar
    column = ir.INETColumn

    __slots__ = ()


# ---------------------------------------------------------------------
any = Any()
null = Null()
boolean = Boolean()
int_ = Integer()
int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
uint_ = UnsignedInteger()
uint8 = UInt8()
uint16 = UInt16()
uint32 = UInt32()
uint64 = UInt64()
float = Float()
halffloat = Halffloat()
float16 = Halffloat()
float32 = Float32()
float64 = Float64()
double = Double()
string = String()
binary = Binary()
date = Date()
time = Time()
timestamp = Timestamp()
interval = Interval()
category = Category()
# geo spatial data type
geometry = GeoSpatial()
geography = GeoSpatial()
point = Point()
linestring = LineString()
polygon = Polygon()
multilinestring = MultiLineString()
multipoint = MultiPoint()
multipolygon = MultiPolygon()
# json
json = JSON()
jsonb = JSONB()
# special string based data type
uuid = UUID()
macaddr = MACADDR()
inet = INET()


_STRING_REGEX = """('[^\n'\\\\]*(?:\\\\.[^\n'\\\\]*)*'|"[^\n"\\\\"]*(?:\\\\.[^\n"\\\\]*)*")"""  # noqa: E501

_SPACES = p.regex(r'\s*', re.MULTILINE)


def spaceless(parser):
    return _SPACES.then(parser).skip(_SPACES)


def spaceless_string(s: str):
    return spaceless(p.string(s, transform=str.lower))


@functools.lru_cache(maxsize=100)
def parse_type(text: str) -> DataType:
    """Parse a type from a [`str`][str] `text`.

    The default `maxsize` parameter for caching is chosen to cache the most
    commonly used types--there are about 30--along with some capacity for less
    common but repeatedly-used complex types.

    Parameters
    ----------
    text
        The type string to parse

    Examples
    --------
    Parse an array type from a string

    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> dt.parse_type("array<int64>")
    Array(value_type=int64, nullable=True)

    You can avoid parsing altogether by constructing objects directly

    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> ty = dt.parse_type("array<int64>")
    >>> ty == dt.Array(dt.int64)
    True
    """
    precision = scale = srid = p.digit.at_least(1).concat().map(int)

    lparen = spaceless_string("(")
    rparen = spaceless_string(")")

    langle = spaceless_string("<")
    rangle = spaceless_string(">")

    comma = spaceless_string(",")
    colon = spaceless_string(":")
    semicolon = spaceless_string(";")

    raw_string = p.regex(_STRING_REGEX).map(ast.literal_eval)

    geotype = spaceless_string("geography") | spaceless_string("geometry")

    @p.generate
    def srid_geotype():
        yield semicolon
        sr = yield srid
        yield colon
        gt = yield geotype
        return (gt, sr)

    @p.generate
    def geotype_part():
        yield colon
        gt = yield geotype
        return (gt, None)

    @p.generate
    def srid_part():
        yield semicolon
        sr = yield srid
        return (None, sr)

    def geotype_parser(name, type):
        name_parser = spaceless_string(name)
        geosubtype_parser = srid_geotype | geotype_part | srid_part

        @p.generate
        def parser():
            yield name_parser
            sr_gt = yield geosubtype_parser.optional()
            return type(*sr_gt) if sr_gt is not None else type()

        return parser

    primitive = (
        spaceless_string("any").result(any)
        | spaceless_string("null").result(null)
        | spaceless_string("boolean").result(boolean)
        | spaceless_string("bool").result(boolean)
        | spaceless_string("int8").result(int8)
        | spaceless_string("int16").result(int16)
        | spaceless_string("int32").result(int32)
        | spaceless_string("int64").result(int64)
        | spaceless_string("uint8").result(uint8)
        | spaceless_string("uint16").result(uint16)
        | spaceless_string("uint32").result(uint32)
        | spaceless_string("uint64").result(uint64)
        | spaceless_string("halffloat").result(halffloat)
        | spaceless_string("double").result(double)
        | spaceless_string("float16").result(float16)
        | spaceless_string("float32").result(float32)
        | spaceless_string("float64").result(float64)
        | spaceless_string("float").result(float)
        | spaceless_string("string").result(string)
        | spaceless_string("binary").result(binary)
        | spaceless_string("timestamp").result(Timestamp())
        | spaceless_string("time").result(time)
        | spaceless_string("date").result(date)
        | spaceless_string("category").result(category)
        | spaceless_string("geometry").result(GeoSpatial(geotype='geometry'))
        | spaceless_string("geography").result(GeoSpatial(geotype='geography'))
        | geotype_parser("linestring", LineString)
        | geotype_parser("polygon", Polygon)
        | geotype_parser("point", Point)
        | geotype_parser("multilinestring", MultiLineString)
        | geotype_parser("multipolygon", MultiPolygon)
        | geotype_parser("multipoint", MultiPoint)
    )

    @p.generate
    def varchar_or_char():
        yield p.alt(
            spaceless_string("varchar"), spaceless_string("char")
        ).then(
            lparen.then(p.digit.at_least(1).concat()).skip(rparen).optional()
        )
        return String()

    @p.generate
    def decimal():
        yield spaceless_string("decimal")
        prec_scale = (
            yield lparen.then(
                p.seq(precision.skip(comma), scale).combine(
                    lambda prec, scale: (prec, scale)
                )
            )
            .skip(rparen)
            .optional()
        ) or (9, 0)
        return Decimal(*prec_scale)

    @p.generate
    def parened_string():
        yield lparen
        s = yield raw_string
        yield rparen
        return s

    @p.generate
    def timestamp():
        yield spaceless_string("timestamp")
        tz = yield parened_string
        return Timestamp(tz)

    @p.generate
    def angle_type():
        yield langle
        value_type = yield ty
        yield rangle
        return value_type

    @p.generate
    def interval():
        yield spaceless_string("interval")
        value_type = yield angle_type.optional()
        un = yield parened_string.optional()
        return Interval(
            value_type=value_type, unit=un if un is not None else 's'
        )

    @p.generate
    def array():
        yield spaceless_string("array")
        value_type = yield angle_type
        return Array(value_type)

    @p.generate
    def set():
        yield spaceless_string("set")
        value_type = yield angle_type
        return Set(value_type)

    @p.generate
    def map():
        yield spaceless_string("map")
        yield langle
        key_type = yield primitive
        yield comma
        value_type = yield ty
        yield rangle
        return Map(key_type, value_type)

    field = spaceless(p.regex("[a-zA-Z_][a-zA-Z_0-9]*"))

    @p.generate
    def struct():
        yield spaceless_string("struct")
        yield langle
        field_names_types = yield (
            p.seq(field.skip(colon), ty)
            .combine(lambda field, ty: (field, ty))
            .sep_by(comma)
        )
        yield rangle
        return Struct.from_tuples(field_names_types)

    ty = (
        timestamp
        | primitive
        | decimal
        | varchar_or_char
        | interval
        | array
        | set
        | map
        | struct
        | spaceless_string("jsonb").result(jsonb)
        | spaceless_string("json").result(json)
        | spaceless_string("uuid").result(uuid)
        | spaceless_string("macaddr").result(macaddr)
        | spaceless_string("inet").result(inet)
        | spaceless_string("geography").result(geography)
        | spaceless_string("geometry").result(geometry)
    )

    return ty.parse(text)


dtype = Dispatcher('dtype')

validate_type = dtype


def _get_timedelta_units(
    timedelta: datetime.timedelta | pd.Timedelta,
) -> list[str]:
    # pandas Timedelta has more granularity
    if isinstance(timedelta, pd.Timedelta):
        unit_fields = timedelta.components._fields
        base_object = timedelta.components
    # datetime.timedelta only stores days, seconds, and microseconds internally
    else:
        unit_fields = ['days', 'seconds', 'microseconds']
        base_object = timedelta

    time_units = [
        field for field in unit_fields if getattr(base_object, field) > 0
    ]
    return time_units


@dtype.register(object)
def default(value, **kwargs) -> DataType:
    raise com.IbisTypeError(f'Value {value!r} is not a valid datatype')


@dtype.register(DataType)
def from_ibis_dtype(value: DataType) -> DataType:
    return value


@dtype.register(str)
def from_string(value: str) -> DataType:
    try:
        return parse_type(value)
    except SyntaxError:
        raise com.IbisTypeError(f'{value!r} cannot be parsed as a datatype')


@dtype.register(list)
def from_list(values: list[typing.Any]) -> Array:
    if not values:
        return Array(null)
    return Array(highest_precedence(map(dtype, values)))


@dtype.register(collections.abc.Set)
def from_set(values: set) -> Set:
    if not values:
        return Set(null)
    return Set(highest_precedence(map(dtype, values)))


infer = Dispatcher('infer')


def higher_precedence(left: DataType, right: DataType) -> DataType:
    if castable(left, right, upcast=True):
        return right
    elif castable(right, left, upcast=True):
        return left

    raise com.IbisTypeError(
        f'Cannot compute precedence for {left} and {right} types'
    )


def highest_precedence(dtypes: Iterator[DataType]) -> DataType:
    """Compute the highest precedence of `dtypes`."""
    return functools.reduce(higher_precedence, dtypes)


@infer.register(object)
def infer_dtype_default(value: typing.Any) -> DataType:
    """Default implementation of :func:`~ibis.expr.datatypes.infer`."""
    raise com.InputTypeError(value)


@infer.register(collections.OrderedDict)
def infer_struct(value: Mapping[str, typing.Any]) -> Struct:
    """Infer the :class:`~ibis.expr.datatypes.Struct` type of `value`."""
    if not value:
        raise TypeError('Empty struct type not supported')
    return Struct(list(value.keys()), list(map(infer, value.values())))


@infer.register(collections.abc.Mapping)
def infer_map(value: Mapping[typing.Any, typing.Any]) -> Map:
    """Infer the :class:`~ibis.expr.datatypes.Map` type of `value`."""
    if not value:
        return Map(null, null)
    return Map(
        highest_precedence(map(infer, value.keys())),
        highest_precedence(map(infer, value.values())),
    )


@infer.register(list)
def infer_list(values: list[typing.Any]) -> Array:
    """Infer the :class:`~ibis.expr.datatypes.Array` type of `values`."""
    if not values:
        return Array(null)
    return Array(highest_precedence(map(infer, values)))


@infer.register((set, frozenset))
def infer_set(values: set) -> Set:
    """Infer the :class:`~ibis.expr.datatypes.Set` type of `values`."""
    if not values:
        return Set(null)
    return Set(highest_precedence(map(infer, values)))


@infer.register(datetime.time)
def infer_time(value: datetime.time) -> Time:
    return time


@infer.register(datetime.date)
def infer_date(value: datetime.date) -> Date:
    return date


@infer.register(datetime.datetime)
def infer_timestamp(value: datetime.datetime) -> Timestamp:
    if value.tzinfo:
        return Timestamp(timezone=str(value.tzinfo))
    else:
        return timestamp


@infer.register(datetime.timedelta)
def infer_interval(value: datetime.timedelta) -> Interval:
    time_units = _get_timedelta_units(value)
    # we can attempt a conversion in the simplest case, i.e. there is exactly
    # one unit (e.g. pd.Timedelta('2 days') vs. pd.Timedelta('2 days 3 hours')
    if len(time_units) == 1:
        unit = time_units[0]
        return Interval(unit)
    else:
        return interval


@infer.register(str)
def infer_string(value: str) -> String:
    return string


@infer.register(bytes)
def infer_bytes(value: bytes) -> Binary:
    return binary


@infer.register(builtins.float)
def infer_floating(value: builtins.float) -> Double:
    return double


@infer.register(int)
def infer_integer(value: int, allow_overflow: bool = False) -> Integer:
    for dtype in (int8, int16, int32, int64):
        if dtype.bounds.lower <= value <= dtype.bounds.upper:
            return dtype

    if not allow_overflow:
        raise OverflowError(value)

    return int64


@infer.register(enum.Enum)
def infer_enum(value: enum.Enum) -> Enum:
    return Enum(
        infer(value.name),
        infer(value.value),
    )


@infer.register(bool)
def infer_boolean(value: bool) -> Boolean:
    return boolean


@infer.register((type(None), Null))
def infer_null(value: Null | None) -> Null:
    return null


if IS_SHAPELY_AVAILABLE:

    @infer.register(shapely.geometry.Point)
    def infer_shapely_point(value: shapely.geometry.Point) -> Point:
        return point

    @infer.register(shapely.geometry.LineString)
    def infer_shapely_linestring(
        value: shapely.geometry.LineString,
    ) -> LineString:
        return linestring

    @infer.register(shapely.geometry.Polygon)
    def infer_shapely_polygon(value: shapely.geometry.Polygon) -> Polygon:
        return polygon

    @infer.register(shapely.geometry.MultiLineString)
    def infer_shapely_multilinestring(
        value: shapely.geometry.MultiLineString,
    ) -> MultiLineString:
        return multilinestring

    @infer.register(shapely.geometry.MultiPoint)
    def infer_shapely_multipoint(
        value: shapely.geometry.MultiPoint,
    ) -> MultiPoint:
        return multipoint

    @infer.register(shapely.geometry.MultiPolygon)
    def infer_shapely_multipolygon(
        value: shapely.geometry.MultiPolygon,
    ) -> MultiPolygon:
        return multipolygon


castable = Dispatcher('castable')


@castable.register(DataType, DataType)
def can_cast_subtype(source: DataType, target: DataType, **kwargs) -> bool:
    return isinstance(target, source.__class__)


@castable.register(Any, DataType)
@castable.register(DataType, Any)
@castable.register(Any, Any)
@castable.register(Null, Any)
@castable.register(Integer, Category)
@castable.register(Integer, (Floating, Decimal))
@castable.register(Floating, Decimal)
@castable.register((Date, Timestamp), (Date, Timestamp))
def can_cast_any(source: DataType, target: DataType, **kwargs) -> bool:
    return True


@castable.register(Null, DataType)
def can_cast_null(source: DataType, target: DataType, **kwargs) -> bool:
    return target.nullable


Integral = TypeVar('Integral', SignedInteger, UnsignedInteger)


@castable.register(SignedInteger, UnsignedInteger)
@castable.register(UnsignedInteger, SignedInteger)
def can_cast_to_differently_signed_integer_type(
    source: Integral, target: Integral, value: int | None = None, **kwargs
) -> bool:
    if value is None:
        return False
    bounds = target.bounds
    return bounds.lower <= value <= bounds.upper


@castable.register(SignedInteger, SignedInteger)
@castable.register(UnsignedInteger, UnsignedInteger)
def can_cast_integers(source: Integral, target: Integral, **kwargs) -> bool:
    return target._nbytes >= source._nbytes


@castable.register(Floating, Floating)
def can_cast_floats(
    source: Floating, target: Floating, upcast: bool = False, **kwargs
) -> bool:
    if upcast:
        return target._nbytes >= source._nbytes

    # double -> float must be allowed because
    # float literals are inferred as doubles
    return True


@castable.register(Decimal, Decimal)
def can_cast_decimals(source: Decimal, target: Decimal, **kwargs) -> bool:
    return (
        target.precision >= source.precision and target.scale >= source.scale
    )


@castable.register(Interval, Interval)
def can_cast_intervals(source: Interval, target: Interval, **kwargs) -> bool:
    return source.unit == target.unit and castable(
        source.value_type, target.value_type
    )


@castable.register(Integer, Boolean)
def can_cast_integer_to_boolean(
    source: Integer, target: Boolean, value: int | None = None, **kwargs
) -> bool:
    return value is not None and (value == 0 or value == 1)


@castable.register(Integer, Interval)
def can_cast_integer_to_interval(
    source: Interval, target: Interval, **kwargs
) -> bool:
    return castable(source, target.value_type)


@castable.register(String, (Date, Time, Timestamp))
def can_cast_string_to_temporal(
    source: String,
    target: Date | Time | Timestamp,
    value: str | None = None,
    **kwargs,
) -> bool:
    if value is None:
        return False
    try:
        pd.Timestamp(value)
    except ValueError:
        return False
    else:
        return True


Collection = TypeVar('Collection', Array, Set)


@castable.register(Map, Map)
def can_cast_map(source, target, **kwargs):
    return castable(source.key_type, target.key_type) and castable(
        source.value_type, target.value_type
    )


@castable.register(Struct, Struct)
def can_cast_struct(source, target, **kwargs):
    source_pairs = source.pairs
    target_pairs = target.pairs
    for name in {*source.names, *target.names}:
        if name in target_pairs:
            if not castable(source_pairs[name], target_pairs[name]):
                return False
    return True


@castable.register(Array, Array)
@castable.register(Set, Set)
def can_cast_variadic(
    source: Collection, target: Collection, **kwargs
) -> bool:
    return castable(source.value_type, target.value_type)


@castable.register(JSON, JSON)
def can_cast_json(source, target, **kwargs):
    return True


@castable.register(JSONB, JSONB)
def can_cast_jsonb(source, target, **kwargs):
    return True


# geo spatial data type
# cast between same type, used to cast from/to geometry and geography
GEO_TYPES = (
    Point,
    LineString,
    Polygon,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
)


@castable.register(Array, GEO_TYPES)
@castable.register(GEO_TYPES, Geometry)
@castable.register(GEO_TYPES, Geography)
def can_cast_geospatial(source, target, **kwargs):
    return True


@castable.register(UUID, UUID)
@castable.register(UUID, String)
@castable.register(String, UUID)
@castable.register(MACADDR, MACADDR)
@castable.register(INET, INET)
def can_cast_special_string(source, target, **kwargs):
    return True


def cast(source: str | DataType, target: str | DataType, **kwargs) -> DataType:
    """Attempts to implicitly cast from source dtype to target dtype"""
    source, result_target = dtype(source), dtype(target)

    if not castable(source, result_target, **kwargs):
        raise com.IbisTypeError(
            'Datatype {} cannot be implicitly '
            'casted to {}'.format(source, result_target)
        )
    return result_target


same_kind = Dispatcher(
    'same_kind',
    doc="""\
Compute whether two :class:`~ibis.expr.datatypes.DataType` instances are the
same kind.

Parameters
----------
a : DataType
b : DataType

Returns
-------
bool
    Whether two :class:`~ibis.expr.datatypes.DataType` instances are the same
    kind.
""",
)


@same_kind.register(DataType, DataType)
def same_kind_default(a: DataType, b: DataType) -> bool:
    """Return whether `a` is exactly equiavlent to `b`"""
    return a.equals(b)


Numeric = TypeVar('Numeric', Integer, Floating)


@same_kind.register(Integer, Integer)
@same_kind.register(Floating, Floating)
def same_kind_numeric(a: Numeric, b: Numeric) -> bool:
    """Return ``True``."""
    return True


@same_kind.register(DataType, Null)
def same_kind_right_null(a: DataType, _: Null) -> bool:
    """Return whether `a` is nullable."""
    return a.nullable


@same_kind.register(Null, DataType)
def same_kind_left_null(_: Null, b: DataType) -> bool:
    """Return whether `b` is nullable."""
    return b.nullable


@same_kind.register(Null, Null)
def same_kind_both_null(a: Null, b: Null) -> bool:
    """Return ``True``."""
    return True


_normalize = Dispatcher(
    "_normalize",
    doc="""\
Ensure that the Python type underlying an
:class:`~ibis.expr.operations.generic.Literal` resolves to a single acceptable
type regardless of the input value.

Parameters
----------
typ : DataType
value :

Returns
-------
value
    the input ``value`` normalized to the expected type
""",
)


@_normalize.register(DataType, object)
def _normalize_default(typ: DataType, value: object) -> object:
    return value


@_normalize.register(Integer, (int, builtins.float, np.integer, np.floating))
def _int(typ: Integer, value: builtins.float) -> builtins.float:
    return int(value)


@_normalize.register(Floating, (int, builtins.float, np.integer, np.floating))
def _float(typ: Floating, value: builtins.float) -> builtins.float:
    return builtins.float(value)


@_normalize.register(UUID, str)
def _str_to_uuid(typ: UUID, value: str) -> _uuid.UUID:
    return _uuid.UUID(value)


@_normalize.register(String, _uuid.UUID)
def _uuid_to_str(typ: String, value: _uuid.UUID) -> str:
    return str(value)


@_normalize.register(Decimal, int)
def _int_to_decimal(typ: Decimal, value: int) -> decimal.Decimal:
    return decimal.Decimal(value).scaleb(-typ.scale)


@_normalize.register(Array, (tuple, list, np.ndarray))
def _array_to_tuple(typ: Array, values: Sequence) -> tuple:
    return tuple(_normalize(typ.value_type, item) for item in values)


@_normalize.register(Set, (set, frozenset))
def _set_to_frozenset(typ: Set, values: AbstractSet) -> frozenset:
    return frozenset(_normalize(typ.value_type, item) for item in values)


@_normalize.register(Map, dict)
def _map_to_frozendict(typ: Map, values: Mapping) -> decimal.Decimal:
    values = {k: _normalize(typ.value_type, v) for k, v in values.items()}
    return util.frozendict(values)


@_normalize.register(Struct, dict)
def _struct_to_frozendict(typ: Struct, values: Mapping) -> decimal.Decimal:
    value_types = typ.pairs
    values = {
        k: _normalize(typ[k], v) for k, v in values.items() if k in value_types
    }
    return util.frozendict(values)


@_normalize.register(Point, (tuple, list))
def _point_to_tuple(typ: Point, values: Sequence) -> tuple:
    return tuple(_normalize(float64, item) for item in values)


@_normalize.register((LineString, MultiPoint), (tuple, list))
def _linestring_to_tuple(typ: LineString, values: Sequence) -> tuple:
    return tuple(_normalize(point, item) for item in values)


@_normalize.register((Polygon, MultiLineString), (tuple, list))
def _polygon_to_tuple(typ: Polygon, values: Sequence) -> tuple:
    return tuple(_normalize(linestring, item) for item in values)


@_normalize.register(MultiPolygon, (tuple, list))
def _multipolygon_to_tuple(typ: MultiPolygon, values: Sequence) -> tuple:
    return tuple(_normalize(polygon, item) for item in values)


class _WellKnownText(NamedTuple):
    text: str


if IS_SHAPELY_AVAILABLE:
    import shapely.geometry as geom

    @_normalize.register(GeoSpatial, geom.base.BaseGeometry)
    def _geom_to_wkt(
        typ: GeoSpatial,
        base_geom: geom.base.BaseGeometry,
    ) -> _WellKnownText:
        return _WellKnownText(base_geom.wkt)
