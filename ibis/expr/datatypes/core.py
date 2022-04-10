from __future__ import annotations

import ast
import collections
import datetime
import enum
import functools
import numbers
import re
import typing
import uuid as _uuid
from decimal import Decimal as PythonDecimal
from typing import (
    AbstractSet,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Sequence,
    TypeVar,
)

import numpy as np
import pandas as pd
import parsy as p
from multipledispatch import Dispatcher
from public import public

from ibis.common.exceptions import IbisTypeError, InputTypeError
from ibis.common.grounds import Annotable, Comparable, Singleton
from ibis.common.validators import (
    compose_of,
    instance_of,
    isin,
    map_to,
    optional,
    tuple_of,
    validator,
)
from ibis.expr import types as ir
from ibis.util import frozendict

try:
    import shapely.geometry

    IS_SHAPELY_AVAILABLE = True
except ImportError:
    IS_SHAPELY_AVAILABLE = False


dtype = Dispatcher('dtype')

validate_type = dtype


@dtype.register(object)
def default(value, **kwargs) -> DataType:
    raise IbisTypeError(f'Value {value!r} is not a valid datatype')


@dtype.register(str)
def from_string(value: str) -> DataType:
    try:
        return parse_type(value)
    except SyntaxError:
        raise IbisTypeError(f'{value!r} cannot be parsed as a datatype')


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


@validator
def datatype(arg, **kwargs):
    return dtype(arg)


@public
class DataType(Annotable, Comparable):
    """Base class for all data types.

    [`DataType`][ibis.expr.datatypes.DataType] instances are
    immutable.
    """

    nullable = optional(instance_of(bool), default=True)

    def __call__(self, nullable: bool = True) -> DataType:
        if nullable is not True and nullable is not False:
            raise TypeError(
                "__call__ only accepts the 'nullable' argument. "
                "Please construct a new instance of the type to change the "
                "values of the attributes."
            )
        kwargs = dict(zip(self.argnames, self.args))
        kwargs["nullable"] = nullable
        return self.__class__(**kwargs)

    @property
    def _pretty_piece(self) -> str:
        return ""

    @property
    def name(self) -> str:
        """Return the name of the data type."""
        return self.__class__.__name__

    def __str__(self) -> str:
        prefix = "!" * (not self.nullable)
        return f"{prefix}{self.name.lower()}{self._pretty_piece}"

    def __equals__(
        self,
        other: typing.Any,
    ) -> bool:
        return self.args == other.args

    def equals(self, other):
        if not isinstance(other, DataType):
            raise TypeError(
                "invalid equality comparison between DataType and "
                f"{type(other)}"
            )
        return super().__cached_equals__(other)

    def castable(self, target, **kwargs):
        """Return whether this data type is castable to `target`."""
        return castable(self, target, **kwargs)

    def cast(self, target, **kwargs):
        """Cast this data type to `target`."""
        return cast(self, target, **kwargs)


@dtype.register(DataType)
def from_ibis_dtype(value: DataType) -> DataType:
    return value


@public
class Any(DataType):
    """Values of any type."""


@public
class Primitive(Singleton, DataType):
    """Values with known size."""


@public
class Null(Singleton, DataType):
    """Null values."""

    scalar = ir.NullScalar
    column = ir.NullColumn


@public
class Variadic(DataType):
    """Values with unknown size."""


@public
class Boolean(Primitive):
    """[`True`][True] or [`False`][False] values."""

    scalar = ir.BooleanScalar
    column = ir.BooleanColumn


@public
class Bounds(NamedTuple):
    """The lower and upper bound of a fixed-size value."""

    lower: int
    upper: int


@public
class Integer(Primitive):
    """Integer values."""

    scalar = ir.IntegerScalar
    column = ir.IntegerColumn

    @property
    def _nbytes(self) -> int:
        """Return the number of bytes used to store values of this type."""
        raise TypeError(
            "Cannot determine the size in bytes of an abstract integer type."
        )


@public
class String(Singleton, Variadic):
    """A type representing a string.

    Notes
    -----
    Because of differences in the way different backends handle strings, we
    cannot assume that strings are UTF-8 encoded.
    """

    scalar = ir.StringScalar
    column = ir.StringColumn


@public
class Binary(Singleton, Variadic):
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


@public
class Date(Primitive):
    """Date values."""

    scalar = ir.DateScalar
    column = ir.DateColumn


@public
class Time(Primitive):
    """Time values."""

    scalar = ir.TimeScalar
    column = ir.TimeColumn


@public
class Timestamp(DataType):
    """Timestamp values."""

    timezone = optional(instance_of(str))
    """The timezone of values of this type."""

    scalar = ir.TimestampScalar
    column = ir.TimestampColumn

    @property
    def _pretty_piece(self) -> str:
        if (timezone := self.timezone) is not None:
            return f"({timezone!r})"
        return ""


@public
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


@public
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


@public
class Floating(Primitive):
    """Floating point values."""

    scalar = ir.FloatingScalar
    column = ir.FloatingColumn

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


@public
class Int8(SignedInteger):
    """Signed 8-bit integers."""

    _nbytes = 1


@public
class Int16(SignedInteger):
    """Signed 16-bit integers."""

    _nbytes = 2


@public
class Int32(SignedInteger):
    """Signed 32-bit integers."""

    _nbytes = 4


@public
class Int64(SignedInteger):
    """Signed 64-bit integers."""

    _nbytes = 8


@public
class UInt8(UnsignedInteger):
    """Unsigned 8-bit integers."""

    _nbytes = 1


@public
class UInt16(UnsignedInteger):
    """Unsigned 16-bit integers."""

    _nbytes = 2


@public
class UInt32(UnsignedInteger):
    """Unsigned 32-bit integers."""

    _nbytes = 4


@public
class UInt64(UnsignedInteger):
    """Unsigned 64-bit integers."""

    _nbytes = 8


@public
class Float16(Floating):
    """16-bit floating point numbers."""

    _nbytes = 2


@public
class Float32(Floating):
    """32-bit floating point numbers."""

    _nbytes = 4


@public
class Float64(Floating):
    """64-bit floating point numbers."""

    _nbytes = 8


@public
class Decimal(DataType):
    """Fixed-precision decimal values."""

    precision = optional(instance_of(int))
    """The number of decimal places values of this type can hold."""

    scale = optional(instance_of(int))
    """The number of values after the decimal point."""

    scalar = ir.DecimalScalar
    column = ir.DecimalColumn

    def __init__(
        self,
        precision: int | None = None,
        scale: int | None = None,
        **kwargs: Any,
    ) -> None:
        if precision is not None:
            if not isinstance(precision, numbers.Integral):
                raise TypeError(
                    "Decimal type precision must be an integer; "
                    f"got {type(precision)}"
                )
            if precision < 0:
                raise ValueError('Decimal type precision cannot be negative')
            if not precision:
                raise ValueError('Decimal type precision cannot be zero')
        if scale is not None:
            if not isinstance(scale, numbers.Integral):
                raise TypeError('Decimal type scale must be an integer')
            if scale < 0:
                raise ValueError('Decimal type scale cannot be negative')
            if precision is not None and precision < scale:
                raise ValueError(
                    'Decimal type precision must be greater than or equal to '
                    'scale. Got precision={:d} and scale={:d}'.format(
                        precision, scale
                    )
                )
        super().__init__(precision=precision, scale=scale, **kwargs)

    @property
    def largest(self):
        """Return the largest type of decimal."""
        return self.__class__(precision=None, scale=None)

    @property
    def _pretty_piece(self) -> str:
        args = []

        if (precision := self.precision) is not None:
            args.append(f"prec={precision:d}")

        if (scale := self.scale) is not None:
            args.append(f"scale={scale:d}")

        if not args:
            return ""

        return f"({', '.join(args)})"


@public
class Interval(DataType):
    """Interval values."""

    unit = optional(
        map_to(
            {
                'days': 'D',
                'hours': 'h',
                'minutes': 'm',
                'seconds': 's',
                'milliseconds': 'ms',
                'microseconds': 'us',
                'nanoseconds': 'ns',
                'Y': 'Y',
                'Q': 'Q',
                'M': 'M',
                'W': 'W',
                'D': 'D',
                'h': 'h',
                'm': 'm',
                's': 's',
                'ms': 'ms',
                'us': 'us',
                'ns': 'ns',
            }
        ),
        default="s",
    )
    """The time unit of the interval."""

    value_type = optional(
        compose_of([datatype, instance_of(Integer)]), default=Int32()
    )
    """The underlying type of the stored values."""

    scalar = ir.IntervalScalar
    column = ir.IntervalColumn

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

    # TODO(kszucs): assert that the nullability if the value_type is equal
    # to the interval's nullability

    @property
    def bounds(self):
        return self.value_type.bounds

    @property
    def resolution(self):
        """The interval unit's name."""
        return self._units[self.unit]

    @property
    def _pretty_piece(self) -> str:
        return f"<{self.value_type}>(unit={self.unit!r})"


@public
class Category(DataType):
    cardinality = optional(instance_of(int))

    scalar = ir.CategoryScalar
    column = ir.CategoryColumn

    def __repr__(self):
        if self.cardinality is not None:
            cardinality = repr(self.cardinality)
        else:
            cardinality = "unknown"
        return f"{self.name}(cardinality={cardinality})"

    def to_integer_type(self):
        if self.cardinality is None:
            return int64
        else:
            return infer(self.cardinality)


@public
class Struct(DataType):
    """Structured values."""

    names = tuple_of(instance_of(str))
    types = tuple_of(datatype)

    scalar = ir.StructScalar
    column = ir.StructColumn

    @classmethod
    def from_tuples(
        cls,
        pairs: Iterable[tuple[str, str | DataType]],
        nullable: bool = True,
    ) -> Struct:
        """Construct a `Struct` type from pairs.

        Parameters
        ----------
        pairs
            An iterable of pairs of field name and type

        Returns
        -------
        Struct
            Struct data type instance
        """
        names, types = zip(*pairs)
        return cls(names, types, nullable=nullable)

    @classmethod
    def from_dict(
        cls,
        pairs: Mapping[str, str | DataType],
        nullable: bool = True,
    ) -> Struct:
        """Construct a `Struct` type from a [`dict`][dict].

        Parameters
        ----------
        pairs
            A [`dict`][dict] of `field: type`

        Returns
        -------
        Struct
            Struct data type instance
        """
        names, types = pairs.keys(), pairs.values()
        return cls(names, types, nullable=nullable)

    @property
    def pairs(self) -> Mapping[str, DataType]:
        """Return a mapping from names to data type instances.

        Returns
        -------
        Mapping[str, DataType]
            Mapping of field name to data type
        """
        return dict(zip(self.names, self.types))

    def __getitem__(self, key: str) -> DataType:
        return self.pairs[key]

    def __repr__(self) -> str:
        return '{}({}, nullable={})'.format(
            self.name, list(self.pairs.items()), self.nullable
        )

    @property
    def _pretty_piece(self) -> str:
        pairs = ", ".join(map("{}: {}".format, self.names, self.types))
        return f"<{pairs}>"


@public
class Array(Variadic):
    """Array values."""

    value_type = datatype

    scalar = ir.ArrayScalar
    column = ir.ArrayColumn

    @property
    def _pretty_piece(self) -> str:
        return f"<{self.value_type}>"


@public
class Set(Variadic):
    """Set values."""

    value_type = datatype

    scalar = ir.SetScalar
    column = ir.SetColumn

    @property
    def _pretty_piece(self) -> str:
        return f"<{self.value_type}>"


@public
class Enum(DataType):
    """Enumeration values."""

    rep_type = datatype
    value_type = datatype

    scalar = ir.EnumScalar
    column = ir.EnumColumn


@public
class Map(Variadic):
    """Associative array values."""

    key_type = datatype
    value_type = datatype

    scalar = ir.MapScalar
    column = ir.MapColumn

    @property
    def _pretty_piece(self) -> str:
        return f"<{self.key_type}, {self.value_type}>"


@public
class JSON(String):
    """JSON values."""

    scalar = ir.JSONScalar
    column = ir.JSONColumn


@public
class JSONB(Binary):
    """JSON data stored in a binary representation.

    This representation eliminates whitespace, duplicate keys, and does not
    preserve key ordering.
    """

    scalar = ir.JSONBScalar
    column = ir.JSONBColumn


@public
class GeoSpatial(DataType):
    """Geospatial values."""

    geotype = optional(isin({"geography", "geometry"}))
    """The specific geospatial type"""

    srid = optional(instance_of(int))
    """The spatial reference identifier."""

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    @property
    def _pretty_piece(self) -> str:
        piece = ""
        if self.geotype is not None:
            piece += f":{self.geotype}"
        if self.srid is not None:
            piece += f";{self.srid}"
        return piece


@public
class Geometry(GeoSpatial):
    """Geometry values."""

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    def __init__(self, geotype, **kwargs):
        super().__init__(geotype="geometry", **kwargs)


@public
class Geography(GeoSpatial):
    """Geography values."""

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    def __init__(self, geotype, **kwargs):
        super().__init__(geotype="geography", **kwargs)


@public
class Point(GeoSpatial):
    """A point described by two coordinates."""

    scalar = ir.PointScalar
    column = ir.PointColumn


@public
class LineString(GeoSpatial):
    """A sequence of 2 or more points."""

    scalar = ir.LineStringScalar
    column = ir.LineStringColumn


@public
class Polygon(GeoSpatial):
    """A set of one or more closed line strings.

    The first line string represents the shape (external ring) and the rest
    represent holes in that shape (internal rings).
    """

    scalar = ir.PolygonScalar
    column = ir.PolygonColumn


@public
class MultiLineString(GeoSpatial):
    """A set of one or more line strings."""

    scalar = ir.MultiLineStringScalar
    column = ir.MultiLineStringColumn


@public
class MultiPoint(GeoSpatial):
    """A set of one or more points."""

    scalar = ir.MultiPointScalar
    column = ir.MultiPointColumn


@public
class MultiPolygon(GeoSpatial):
    """A set of one or more polygons."""

    scalar = ir.MultiPolygonScalar
    column = ir.MultiPolygonColumn


@public
class UUID(DataType):
    """A 128-bit number used to identify information in computer systems."""

    scalar = ir.UUIDScalar
    column = ir.UUIDColumn


@public
class MACADDR(String):
    """Media Access Control (MAC) address of a network interface."""

    scalar = ir.MACADDRScalar
    column = ir.MACADDRColumn


@public
class INET(String):
    """IP addresses."""

    scalar = ir.INETScalar
    column = ir.INETColumn


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

# ---------------------------------------------------------------------

infer = Dispatcher('infer')
castable = Dispatcher('castable')

# ---------------------------------------------------------------------

any = Any()
null = Null()
boolean = Boolean()
int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
uint8 = UInt8()
uint16 = UInt16()
uint32 = UInt32()
uint64 = UInt64()
float16 = Float16()
float32 = Float32()
float64 = Float64()
string = String()
binary = Binary()
date = Date()
time = Time()
timestamp = Timestamp()
interval = Interval()
category = Category()
# geo spatial data type
geometry = Geometry()
geography = Geography()
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
decimal = Decimal()

public(
    any=any,
    null=null,
    boolean=boolean,
    int8=int8,
    int16=int16,
    int32=int32,
    int64=int64,
    uint8=uint8,
    uint16=uint16,
    uint32=uint32,
    uint64=uint64,
    float16=float16,
    float32=float32,
    float64=float64,
    string=string,
    binary=binary,
    date=date,
    time=time,
    timestamp=timestamp,
    dtype=dtype,
    infer=infer,
    castable=castable,
    same_kind=same_kind,
    interval=interval,
    category=category,
    geometry=geometry,
    geography=geography,
    point=point,
    linestring=linestring,
    polygon=polygon,
    multilinestring=multilinestring,
    multipoint=multipoint,
    multipolygon=multipolygon,
    json=json,
    jsonb=jsonb,
    uuid=uuid,
    macaddr=macaddr,
    inet=inet,
    decimal=decimal,
)

_STRING_REGEX = """('[^\n'\\\\]*(?:\\\\.[^\n'\\\\]*)*'|"[^\n"\\\\"]*(?:\\\\.[^\n"\\\\]*)*")"""  # noqa: E501

_SPACES = p.regex(r'\s*', re.MULTILINE)


def spaceless(parser):
    return _SPACES.then(parser).skip(_SPACES)


def spaceless_string(s: str):
    return spaceless(p.string(s, transform=str.lower))


@public
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
    Array(value_type=Int64(nullable=True), nullable=True)

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
        spaceless_string("boolean").result(boolean)  # docprecated
        | spaceless_string("bool").result(boolean)
        | spaceless_string("int8").result(int8)
        | spaceless_string("int16").result(int16)
        | spaceless_string("int32").result(int32)
        | spaceless_string("int64").result(int64)
        | spaceless_string("uint8").result(uint8)
        | spaceless_string("uint16").result(uint16)
        | spaceless_string("uint32").result(uint32)
        | spaceless_string("uint64").result(uint64)
        | spaceless_string("halffloat").result(float16)  # docprecated
        | spaceless_string("double").result(float64)  # docprecated
        | spaceless_string("float16").result(float16)
        | spaceless_string("float32").result(float32)
        | spaceless_string("float64").result(float64)
        | spaceless_string("float").result(float64)
        | spaceless_string("string").result(string)
        | spaceless_string("binary").result(binary)  # docprecated
        | spaceless_string("bytes").result(binary)
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
        prec, sc = (
            yield lparen.then(
                p.seq(precision.skip(comma), scale).combine(
                    lambda prec, scale: (prec, scale)
                )
            )
            .skip(rparen)
            .optional()
        ) or (None, None)
        return Decimal(precision=prec, scale=sc)

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
        | spaceless_string("int").result(int64)
        | spaceless_string("str").result(string)
    )

    return ty.parse(text)


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

    return [field for field in unit_fields if getattr(base_object, field) > 0]


def higher_precedence(left: DataType, right: DataType) -> DataType:
    if castable(left, right, upcast=True):
        return right
    elif castable(right, left, upcast=True):
        return left

    raise IbisTypeError(
        f'Cannot compute precedence for {left} and {right} types'
    )


@public
def highest_precedence(dtypes: Iterator[DataType]) -> DataType:
    """Compute the highest precedence of `dtypes`."""
    return functools.reduce(higher_precedence, dtypes)


@infer.register(object)
def infer_dtype_default(value: typing.Any) -> DataType:
    """Default implementation of :func:`~ibis.expr.datatypes.infer`."""
    raise InputTypeError(value)


@infer.register(collections.OrderedDict)
def infer_struct(value: Mapping[str, typing.Any]) -> Struct:
    """Infer the [`Struct`][ibis.expr.datatypes.Struct] type of `value`."""
    if not value:
        raise TypeError('Empty struct type not supported')
    return Struct(list(value.keys()), list(map(infer, value.values())))


@infer.register(collections.abc.Mapping)
def infer_map(value: Mapping[typing.Any, typing.Any]) -> Map:
    """Infer the [`Map`][ibis.expr.datatypes.Map] type of `value`."""
    if not value:
        return Map(null, null)
    return Map(
        highest_precedence(map(infer, value.keys())),
        highest_precedence(map(infer, value.values())),
    )


@infer.register((list, tuple))
def infer_list(values: typing.Sequence[typing.Any]) -> Array:
    """Infer the [`Array`][ibis.expr.datatypes.Array] type of `value`."""
    if not values:
        return Array(null)
    return Array(highest_precedence(map(infer, values)))


@infer.register((set, frozenset))
def infer_set(values: set) -> Set:
    """Infer the [`Set`][ibis.expr.datatypes.Set] type of `value`."""
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


@infer.register(float)
def infer_floating(value: float) -> Float64:
    return float64


@infer.register(int)
def infer_integer(value: int) -> Integer:
    for dtype in (int8, int16, int32, int64):
        if dtype.bounds.lower <= value <= dtype.bounds.upper:
            return dtype
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
    target_prec = target.precision
    source_prec = source.precision
    target_sc = target.scale
    source_sc = source.scale
    return (
        target_prec is None
        or (source_prec is not None and target_prec >= source_prec)
    ) and (
        target_sc is None or (source_sc is not None and target_sc >= source_sc)
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
    source: Integer, target: Interval, **kwargs
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
    source, target = dtype(source), dtype(target)

    if not castable(source, target, **kwargs):
        raise IbisTypeError(
            f'Datatype {source} cannot be implicitly casted to {target}'
        )
    return target


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


@_normalize.register(Integer, (int, float, np.integer, np.floating))
def _int(typ: Integer, value: float) -> float:
    return int(value)


@_normalize.register(Floating, (int, float, np.integer, np.floating))
def _float(typ: Floating, value: float) -> float:
    return float(value)


@_normalize.register(UUID, str)
def _str_to_uuid(typ: UUID, value: str) -> _uuid.UUID:
    return _uuid.UUID(value)


@_normalize.register(String, _uuid.UUID)
def _uuid_to_str(typ: String, value: _uuid.UUID) -> str:
    return str(value)


@_normalize.register(Decimal, int)
def _int_to_decimal(typ: Decimal, value: int) -> PythonDecimal:
    return PythonDecimal(value).scaleb(-typ.scale)


@_normalize.register(Array, (tuple, list, np.ndarray))
def _array_to_tuple(typ: Array, values: Sequence) -> tuple:
    return tuple(_normalize(typ.value_type, item) for item in values)


@_normalize.register(Set, (set, frozenset))
def _set_to_frozenset(typ: Set, values: AbstractSet) -> frozenset:
    return frozenset(_normalize(typ.value_type, item) for item in values)


@_normalize.register(Map, dict)
def _map_to_frozendict(typ: Map, values: Mapping) -> PythonDecimal:
    values = {k: _normalize(typ.value_type, v) for k, v in values.items()}
    return frozendict(values)


@_normalize.register(Struct, dict)
def _struct_to_frozendict(typ: Struct, values: Mapping) -> PythonDecimal:
    value_types = typ.pairs
    values = {
        k: _normalize(typ[k], v) for k, v in values.items() if k in value_types
    }
    return frozendict(values)


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
