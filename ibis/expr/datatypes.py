from __future__ import annotations

import ast
import builtins
import collections
import datetime
import enum
import functools
import itertools
import numbers
import re
import typing
import uuid as _uuid
from typing import Iterator, Mapping, NamedTuple, Sequence, TypeVar

import pandas as pd
import parsy as p
import toolz
from multipledispatch import Dispatcher

import ibis.common.exceptions as com
import ibis.expr.types as ir
from ibis import util

IS_SHAPELY_AVAILABLE = False
try:
    import shapely.geometry

    IS_SHAPELY_AVAILABLE = True
except ImportError:
    ...


class DataType:

    __slots__ = ('nullable',)

    def __init__(self, nullable: bool = True, **kwargs) -> None:
        self.nullable = nullable

    def __call__(self, nullable: bool = True) -> DataType:
        if nullable is not True and nullable is not False:
            raise TypeError(
                "__call__ only accepts the 'nullable' argument. "
                "Please construct a new instance of the type to change the "
                "values of the attributes."
            )
        return self._factory(nullable=nullable)

    def _factory(self, nullable: bool = True) -> DataType:
        slots = {
            slot: getattr(self, slot)
            for slot in self.__slots__
            if slot != 'nullable'
        }
        return type(self)(nullable=nullable, **slots)

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        custom_parts = tuple(
            getattr(self, slot)
            for slot in toolz.unique(self.__slots__ + ('nullable',))
        )
        return hash((type(self),) + custom_parts)

    def __repr__(self) -> str:
        return '{}({})'.format(
            self.name,
            ', '.join(
                f'{slot}={getattr(self, slot)!r}'
                for slot in toolz.unique(self.__slots__ + ('nullable',))
            ),
        )

    def __str__(self) -> str:
        return '{}{}'.format(
            self.name.lower(), '[non-nullable]' if not self.nullable else ''
        )

    @property
    def name(self) -> str:
        return type(self).__name__

    def equals(
        self,
        other: DataType,
        cache: Mapping[typing.Any, bool] | None = None,
    ) -> bool:
        if isinstance(other, str):
            raise TypeError(
                'Comparing datatypes to strings is not allowed. Convert '
                '{!r} to the equivalent DataType instance.'.format(other)
            )
        return (
            isinstance(other, type(self))
            and self.nullable == other.nullable
            and self.__slots__ == other.__slots__
            and all(
                getattr(self, slot) == getattr(other, slot)
                for slot in self.__slots__
            )
        )

    def castable(self, target, **kwargs):
        return castable(self, target, **kwargs)

    def cast(self, target, **kwargs):
        return cast(self, target, **kwargs)

    def scalar_type(self):
        return functools.partial(self.scalar, dtype=self)

    def column_type(self):
        return functools.partial(self.column, dtype=self)

    def _literal_value_hash_key(self, value) -> tuple[DataType, typing.Any]:
        """Return a hash for `value`."""
        return self, value


class Any(DataType):
    __slots__ = ()


class Primitive(DataType):
    __slots__ = ()

    def __repr__(self) -> str:
        name = self.name.lower()
        if not self.nullable:
            return f'{name}[non-nullable]'
        return name


class Null(DataType):
    scalar = ir.NullScalar
    column = ir.NullColumn

    __slots__ = ()


class Variadic(DataType):
    __slots__ = ()


class Boolean(Primitive):
    scalar = ir.BooleanScalar
    column = ir.BooleanColumn

    __slots__ = ()


class Bounds(NamedTuple):
    lower: int
    upper: int


class Integer(Primitive):
    scalar = ir.IntegerScalar
    column = ir.IntegerColumn

    __slots__ = ()

    @property
    def _nbytes(self) -> int:
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
    """A type representing a blob of bytes.

    Notes
    -----
    Some databases treat strings and blobs of equally, and some do not. For
    example, Impala doesn't make a distinction between string and binary types
    but PostgreSQL has a TEXT type and a BYTEA type which are distinct types
    that behave differently.
    """

    scalar = ir.BinaryScalar
    column = ir.BinaryColumn

    __slots__ = ()


class Date(Primitive):
    scalar = ir.DateScalar
    column = ir.DateColumn

    __slots__ = ()


class Time(Primitive):
    scalar = ir.TimeScalar
    column = ir.TimeColumn

    __slots__ = ()


class Timestamp(DataType):
    scalar = ir.TimestampScalar
    column = ir.TimestampColumn

    __slots__ = ('timezone',)

    def __init__(
        self, timezone: str | None = None, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.timezone = timezone

    def __str__(self) -> str:
        timezone = self.timezone
        typename = self.name.lower()
        if timezone is None:
            return typename
        return f'{typename}({timezone!r})'


class SignedInteger(Integer):
    @property
    def largest(self):
        return int64

    @property
    def bounds(self):
        exp = self._nbytes * 8 - 1
        upper = (1 << exp) - 1
        return Bounds(lower=~upper, upper=upper)


class UnsignedInteger(Integer):
    @property
    def largest(self):
        return uint64

    @property
    def bounds(self):
        exp = self._nbytes * 8 - 1
        upper = 1 << exp
        return Bounds(lower=0, upper=upper)


class Floating(Primitive):
    scalar = ir.FloatingScalar
    column = ir.FloatingColumn

    __slots__ = ()

    @property
    def largest(self):
        return float64

    @property
    def _nbytes(self) -> int:
        raise TypeError(
            "Cannot determine the size in bytes of an abstract floating "
            "point type."
        )


class Int8(SignedInteger):
    __slots__ = ()
    _nbytes = 1


class Int16(SignedInteger):
    __slots__ = ()
    _nbytes = 2


class Int32(SignedInteger):
    __slots__ = ()
    _nbytes = 4


class Int64(SignedInteger):
    __slots__ = ()
    _nbytes = 8


class UInt8(UnsignedInteger):
    __slots__ = ()
    _nbytes = 1


class UInt16(UnsignedInteger):
    __slots__ = ()
    _nbytes = 2


class UInt32(UnsignedInteger):
    __slots__ = ()
    _nbytes = 4


class UInt64(UnsignedInteger):
    __slots__ = ()
    _nbytes = 8


class Float16(Floating):
    __slots__ = ()
    _nbytes = 2


class Float32(Floating):
    __slots__ = ()
    _nbytes = 4


class Float64(Floating):
    __slots__ = ()
    _nbytes = 8


Halffloat = Float16
Float = Float32
Double = Float64


class Decimal(DataType):
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

    def __str__(self) -> str:
        return '{}({:d}, {:d})'.format(
            self.name.lower(), self.precision, self.scale
        )

    @property
    def largest(self) -> Decimal:
        return Decimal(38, self.scale)


class Interval(DataType):
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

    @property
    def bounds(self):
        return self.value_type.bounds

    @property
    def resolution(self):
        """Unit's name"""
        return self._units[self.unit]

    def __str__(self):
        unit = self.unit
        typename = self.name.lower()
        value_type_name = self.value_type.name.lower()
        return f'{typename}<{value_type_name}>(unit={unit!r})'


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


class Struct(DataType):
    scalar = ir.StructScalar
    column = ir.StructColumn

    __slots__ = 'names', 'types'

    def __init__(
        self, names: list[str], types: list[DataType], nullable: bool = True
    ) -> None:
        """Construct a ``Struct`` type from a `names` and `types`.

        Parameters
        ----------
        names : Sequence[str]
            Sequence of strings indicating the name of each field in the
            struct.
        types : Sequence[Union[str, DataType]]
            Sequence of strings or :class:`~ibis.expr.datatypes.DataType`
            instances, one for each field
        nullable : bool, optional
            Whether the struct can be null
        """
        if not (names and types):
            raise ValueError('names and types must not be empty')
        if len(names) != len(types):
            raise ValueError('names and types must have the same length')

        super().__init__(nullable=nullable)
        self.names = names
        self.types = types

    @classmethod
    def from_tuples(
        cls,
        pairs: Sequence[tuple[str, str | DataType]],
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
        return collections.OrderedDict(zip(self.names, self.types))

    def __getitem__(self, key: str) -> DataType:
        return self.pairs[key]

    def __hash__(self) -> int:
        return hash(
            (type(self), tuple(self.names), tuple(self.types), self.nullable)
        )

    def __repr__(self) -> str:
        return '{}({}, nullable={})'.format(
            self.name, list(self.pairs.items()), self.nullable
        )

    def __str__(self) -> str:
        return '{}<{}>'.format(
            self.name.lower(),
            ', '.join(itertools.starmap('{}: {}'.format, self.pairs.items())),
        )

    def _literal_value_hash_key(self, value):
        return self, _tuplize(value.items())


def _tuplize(values):
    """Recursively convert `values` to a tuple of tuples."""

    def tuplize_iter(values):
        yield from (
            tuple(tuplize_iter(value)) if util.is_iterable(value) else value
            for value in values
        )

    return tuple(tuplize_iter(values))


class Array(Variadic):
    scalar = ir.ArrayScalar
    column = ir.ArrayColumn

    __slots__ = ('value_type',)

    def __init__(
        self, value_type: str | DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.value_type = dtype(value_type)

    def __str__(self) -> str:
        return f'{self.name.lower()}<{self.value_type}>'

    def _literal_value_hash_key(self, value):
        return self, _tuplize(value)


class Set(Variadic):
    scalar = ir.SetScalar
    column = ir.SetColumn

    __slots__ = ('value_type',)

    def __init__(
        self, value_type: str | DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.value_type = dtype(value_type)

    def __str__(self) -> str:
        return f'{self.name.lower()}<{self.value_type}>'


class Enum(DataType):
    scalar = ir.EnumScalar
    column = ir.EnumColumn

    __slots__ = 'rep_type', 'value_type'

    def __init__(
        self, rep_type: DataType, value_type: DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.rep_type = dtype(rep_type)
        self.value_type = dtype(value_type)


class Map(Variadic):
    scalar = ir.MapScalar
    column = ir.MapColumn

    __slots__ = 'key_type', 'value_type'

    def __init__(
        self, key_type: DataType, value_type: DataType, nullable: bool = True
    ) -> None:
        super().__init__(nullable=nullable)
        self.key_type = dtype(key_type)
        self.value_type = dtype(value_type)

    def __str__(self) -> str:
        return '{}<{}, {}>'.format(
            self.name.lower(), self.key_type, self.value_type
        )

    def _literal_value_hash_key(self, value):
        return self, _tuplize(value.items())


class JSON(String):
    """JSON (JavaScript Object Notation) text format."""

    scalar = ir.JSONScalar
    column = ir.JSONColumn


class JSONB(Binary):
    """JSON (JavaScript Object Notation) data stored as a binary
    representation, which eliminates whitespace, duplicate keys,
    and key ordering.
    """

    scalar = ir.JSONBScalar
    column = ir.JSONBColumn


class GeoSpatial(DataType):
    __slots__ = 'geotype', 'srid'

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    def __init__(
        self, geotype: str = None, srid: int = None, nullable: bool = True
    ):
        """Geospatial data type base class

        Parameters
        ----------
        geotype : str
            Specification of geospatial type which could be `geography` or
            `geometry`.
        srid : int
            Spatial Reference System Identifier
        nullable : bool, optional
            Whether the struct can be null
        """
        super().__init__(nullable=nullable)

        if geotype not in (None, 'geometry', 'geography'):
            raise ValueError(
                'The `geotype` parameter should be `geometry` or `geography`'
            )

        self.geotype = geotype
        self.srid = srid

    def __str__(self) -> str:
        geo_op = self.name.lower()
        if self.geotype is not None:
            geo_op += ':' + self.geotype
        if self.srid is not None:
            geo_op += ';' + str(self.srid)
        return geo_op

    def _literal_value_hash_key(self, value):
        if IS_SHAPELY_AVAILABLE:
            geo_shapes = (
                shapely.geometry.Point,
                shapely.geometry.LineString,
                shapely.geometry.Polygon,
                shapely.geometry.MultiLineString,
                shapely.geometry.MultiPoint,
                shapely.geometry.MultiPolygon,
            )
            if isinstance(value, geo_shapes):
                return self, value.wkt
        return self, value


class Geometry(GeoSpatial):
    """Geometry is used to cast from geography types."""

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geotype = 'geometry'

    def __str__(self) -> str:
        return self.name.lower()


class Geography(GeoSpatial):
    """Geography is used to cast from geometry types."""

    column = ir.GeoSpatialColumn
    scalar = ir.GeoSpatialScalar

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geotype = 'geography'

    def __str__(self) -> str:
        return self.name.lower()


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
    """A set of one or more rings (closed line strings), with the first
    representing the shape (external ring) and the rest representing holes in
    that shape (internal rings).
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


class UUID(String):
    """A universally unique identifier (UUID) is a 128-bit number used to
    identify information in computer systems.
    """

    scalar = ir.UUIDScalar
    column = ir.UUIDColumn

    __slots__ = ()


class MACADDR(String):
    """Media Access Control (MAC) Address of a network interface."""

    scalar = ir.MACADDRScalar
    column = ir.MACADDRColumn

    __slots__ = ()


class INET(String):
    """IP address type."""

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


def parse_type(text: str) -> DataType:
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
    return isinstance(target, type(source))


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
@castable.register(MACADDR, MACADDR)
@castable.register(INET, INET)
def can_cast_special_string(source, target, **kwargs):
    return True


# @castable.register(Map, Map)
# def can_cast_maps(source, target):
#     return (source.equals(target) or
#             source.equals(Map(null, null)) or
#             source.equals(Map(any, any)))
# TODO cast category


def cast(source: DataType | str, target: DataType | str, **kwargs) -> DataType:
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


@_normalize.register(Floating, (int, builtins.float))
def _float(typ: Floating, value: builtins.float) -> builtins.float:
    return builtins.float(value)


@_normalize.register(UUID, str)
def _str_to_uuid(typ: UUID, value: str) -> _uuid.UUID:
    return _uuid.UUID(value)


@_normalize.register(String, _uuid.UUID)
def _uuid_to_str(typ: String, value: _uuid.UUID) -> str:
    return str(value)


@_normalize.register(UUID, _uuid.UUID)
def _uuid_to_uuid(typ: UUID, value: _uuid.UUID) -> _uuid.UUID:
    """Need this to override _uuid_to_str since dt.UUID is a child of
    dt.String"""
    return value
