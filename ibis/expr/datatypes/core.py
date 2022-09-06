from __future__ import annotations

import numbers
from typing import Any, Iterable, Mapping, NamedTuple

from multipledispatch import Dispatcher
from public import public

import ibis.expr.types as ir
from ibis.common.exceptions import IbisTypeError
from ibis.common.grounds import Annotable, Comparable, Singleton
from ibis.common.validators import (
    all_of,
    instance_of,
    isin,
    map_to,
    optional,
    tuple_of,
    validator,
)

dtype = Dispatcher('dtype')


@dtype.register(object)
def default(value, **kwargs) -> DataType:
    raise IbisTypeError(f'Value {value!r} is not a valid datatype')


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

    # TODO(kszucs): remove it in favor of copy()
    def __call__(self, nullable: bool = True) -> DataType:
        return self.copy(nullable=nullable)

    # TODO(kszucs): remove it, prefer to use Annotable.__repr__ instead
    @property
    def _pretty_piece(self) -> str:
        return ""

    # TODO(kszucs): should remove it, only used internally
    @property
    def name(self) -> str:
        """Return the name of the data type."""
        return self.__class__.__name__

    def __str__(self) -> str:
        prefix = "!" * (not self.nullable)
        return f"{prefix}{self.name.lower()}{self._pretty_piece}"

    def __equals__(self, other: Any) -> bool:
        return self.__args__ == other.__args__

    def equals(self, other):
        if not isinstance(other, DataType):
            raise TypeError(
                "invalid equality comparison between DataType and "
                f"{type(other)}"
            )
        return super().__cached_equals__(other)

    def cast(self, other):
        # TODO(kszucs): remove it or deprecate it?
        from ibis.expr.datatypes.cast import cast

        return cast(self, other)

    def castable(self, other):
        # TODO(kszucs): remove it or deprecate it?
        from ibis.expr.datatypes.cast import castable

        return castable(self, other)


@dtype.register(DataType)
def from_ibis_dtype(value: DataType) -> DataType:
    return value


@public
class Primitive(DataType, Singleton):
    """Values with known size."""


@public
class Null(DataType, Singleton):
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
class String(Variadic, Singleton):
    """A type representing a string.

    Notes
    -----
    Because of differences in the way different backends handle strings, we
    cannot assume that strings are UTF-8 encoded.
    """

    scalar = ir.StringScalar
    column = ir.StringColumn


@public
class Binary(Variadic, Singleton):
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
        return self.__class__(
            precision=max(self.precision, 38)
            if self.precision is not None
            else None,
            scale=max(self.scale, 2) if self.scale is not None else None,
        )

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
        all_of([datatype, instance_of(Integer)]), default=Int32()
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
        from ibis.expr.datatypes.value import infer

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


# ---------------------------------------------------------------------

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
