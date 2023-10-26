from __future__ import annotations

import datetime as pydatetime
import decimal as pydecimal
import numbers
import uuid as pyuuid
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Generic, Literal, NamedTuple, Optional, TypeVar, get_type_hints

import toolz
from public import public
from typing_extensions import Self, get_args, get_origin

from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict, MapSet
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.grounds import Concrete, Singleton
from ibis.common.patterns import Coercible, CoercionError
from ibis.common.temporal import IntervalUnit, TimestampUnit


@lazy_singledispatch
def dtype(value: Any, nullable: bool = True) -> DataType:
    """Create a DataType object.

    Parameters
    ----------
    value
        The object to coerce to an Ibis DataType. Supported inputs include
        strings, python type annotations, numpy dtypes, pandas dtypes, and
        pyarrow types.
    nullable
        Whether the type should be nullable. Defaults to True.

    Examples
    --------
    >>> import ibis
    >>> ibis.dtype("int32")
    Int32(nullable=True)
    >>> ibis.dtype("array<float>")
    Array(value_type=Float64(nullable=True), nullable=True)

    DataType objects may also be created from Python types:

    >>> ibis.dtype(int)
    Int64(nullable=True)
    >>> ibis.dtype(list[float])
    Array(value_type=Float64(nullable=True), nullable=True)

    Or other type systems, like numpy/pandas/pyarrow types:

    >>> import pyarrow as pa
    >>> ibis.dtype(pa.int32())
    Int32(nullable=True)
    """
    if isinstance(value, DataType):
        return value
    else:
        return DataType.from_typehint(value)


@dtype.register(str)
def from_string(value):
    return DataType.from_string(value)


@dtype.register("numpy.dtype")
def from_numpy_dtype(value, nullable=True):
    return DataType.from_numpy(value, nullable)


@dtype.register("pandas.core.dtypes.base.ExtensionDtype")
def from_pandas_extension_dtype(value, nullable=True):
    return DataType.from_pandas(value, nullable)


@dtype.register("pyarrow.lib.DataType")
def from_pyarrow(value, nullable=True):
    return DataType.from_pyarrow(value, nullable)


# lock the dispatcher to prevent new types from being registered
del dtype.register


@public
class DataType(Concrete, Coercible):
    """Base class for all data types.

    Instances are immutable.
    """

    nullable: bool = True

    @property
    @abstractmethod
    def scalar(self):
        ...

    @property
    @abstractmethod
    def column(self):
        ...

    # TODO(kszucs): remove it, prefer to use Annotable.__repr__ instead
    @property
    def _pretty_piece(self) -> str:
        return ""

    # TODO(kszucs): should remove it, only used internally
    @property
    def name(self) -> str:
        """Return the name of the data type."""
        return self.__class__.__name__

    @classmethod
    def __coerce__(cls, value, **kwargs):
        if isinstance(value, cls):
            return value
        try:
            return dtype(value)
        except (TypeError, RuntimeError) as e:
            raise CoercionError("Unable to coerce to a DataType") from e

    def __call__(self, **kwargs):
        return self.copy(**kwargs)

    def __str__(self) -> str:
        prefix = "!" * (not self.nullable)
        return f"{prefix}{self.name.lower()}{self._pretty_piece}"

    def equals(self, other):
        if not isinstance(other, DataType):
            raise TypeError(
                f"invalid equality comparison between DataType and {type(other)}"
            )
        return super().__cached_equals__(other)

    def cast(self, other, **kwargs):
        # TODO(kszucs): remove it or deprecate it?
        from ibis.expr.datatypes.cast import cast

        return cast(self, other, **kwargs)

    def castable(self, other, **kwargs):
        # TODO(kszucs): remove it or deprecate it?
        from ibis.expr.datatypes.cast import castable

        return castable(self, other, **kwargs)

    @classmethod
    def from_string(cls, value) -> Self:
        from ibis.expr.datatypes.parse import parse

        try:
            return parse(value)
        except SyntaxError:
            raise TypeError(f"{value!r} cannot be parsed as a datatype")

    @classmethod
    def from_typehint(cls, typ, nullable=True) -> Self:
        origin_type = get_origin(typ)

        if origin_type is None:
            if isinstance(typ, type):
                if issubclass(typ, Parametric):
                    raise TypeError(
                        f"Cannot construct a parametric {typ.__name__} datatype based "
                        "on the type itself"
                    )
                elif issubclass(typ, DataType):
                    return typ(nullable=nullable)
                elif typ is type(None):
                    return null
                elif issubclass(typ, bool):
                    return Boolean(nullable=nullable)
                elif issubclass(typ, bytes):
                    return Binary(nullable=nullable)
                elif issubclass(typ, str):
                    return String(nullable=nullable)
                elif issubclass(typ, Integral):
                    return Int64(nullable=nullable)
                elif issubclass(typ, Real):
                    return Float64(nullable=nullable)
                elif issubclass(typ, pydecimal.Decimal):
                    return Decimal(nullable=nullable)
                elif issubclass(typ, pydatetime.datetime):
                    return Timestamp(nullable=nullable)
                elif issubclass(typ, pydatetime.date):
                    return Date(nullable=nullable)
                elif issubclass(typ, pydatetime.time):
                    return Time(nullable=nullable)
                elif issubclass(typ, pydatetime.timedelta):
                    return Interval(unit="us", nullable=nullable)
                elif issubclass(typ, pyuuid.UUID):
                    return UUID(nullable=nullable)
                elif annots := get_type_hints(typ):
                    return Struct(toolz.valmap(dtype, annots), nullable=nullable)
                else:
                    raise TypeError(
                        f"Cannot construct an ibis datatype from python type `{typ!r}`"
                    )
            else:
                raise TypeError(
                    f"Cannot construct an ibis datatype from python value `{typ!r}`"
                )
        elif issubclass(origin_type, (Sequence, Array)):
            (value_type,) = map(dtype, get_args(typ))
            return Array(value_type)
        elif issubclass(origin_type, (Mapping, Map)):
            key_type, value_type = map(dtype, get_args(typ))
            return Map(key_type, value_type)
        else:
            raise TypeError(f"Value {typ!r} is not a valid datatype")

    @classmethod
    def from_numpy(cls, numpy_type, nullable=True) -> Self:
        """Return the equivalent ibis datatype."""
        from ibis.formats.numpy import NumpyType

        return NumpyType.to_ibis(numpy_type, nullable=nullable)

    @classmethod
    def from_pandas(cls, pandas_type, nullable=True) -> Self:
        """Return the equivalent ibis datatype."""
        from ibis.formats.pandas import PandasType

        return PandasType.to_ibis(pandas_type, nullable=nullable)

    @classmethod
    def from_pyarrow(cls, arrow_type, nullable=True) -> Self:
        """Return the equivalent ibis datatype."""
        from ibis.formats.pyarrow import PyArrowType

        return PyArrowType.to_ibis(arrow_type, nullable=nullable)

    @classmethod
    def from_dask(cls, dask_type, nullable=True) -> Self:
        """Return the equivalent ibis datatype."""
        return cls.from_pandas(dask_type, nullable=nullable)

    def to_numpy(self):
        """Return the equivalent numpy datatype."""
        from ibis.formats.numpy import NumpyFormat

        return NumpyFormat.from_dtype(self)

    def to_pandas(self):
        """Return the equivalent pandas datatype."""
        from ibis.formats.pandas import PandasType

        return PandasType.from_ibis(self)

    def to_pyarrow(self):
        """Return the equivalent pyarrow datatype."""
        from ibis.formats.pyarrow import PyArrowType

        return PyArrowType.from_ibis(self)

    def to_dask(self):
        """Return the equivalent dask datatype."""
        return self.to_pandas()

    def is_array(self) -> bool:
        """Return True if an instance of an Array type."""
        return isinstance(self, Array)

    def is_binary(self) -> bool:
        """Return True if an instance of a Binary type."""
        return isinstance(self, Binary)

    def is_boolean(self) -> bool:
        """Return True if an instance of a Boolean type."""
        return isinstance(self, Boolean)

    def is_date(self) -> bool:
        """Return True if an instance of a Date type."""
        return isinstance(self, Date)

    def is_decimal(self) -> bool:
        """Return True if an instance of a Decimal type."""
        return isinstance(self, Decimal)

    def is_enum(self) -> bool:
        """Return True if an instance of an Enum type."""
        return isinstance(self, Enum)

    def is_float16(self) -> bool:
        """Return True if an instance of a Float16 type."""
        return isinstance(self, Float16)

    def is_float32(self) -> bool:
        """Return True if an instance of a Float32 type."""
        return isinstance(self, Float32)

    def is_float64(self) -> bool:
        """Return True if an instance of a Float64 type."""
        return isinstance(self, Float64)

    def is_floating(self) -> bool:
        """Return True if an instance of any Floating type."""
        return isinstance(self, Floating)

    def is_geospatial(self) -> bool:
        """Return True if an instance of a Geospatial type."""
        return isinstance(self, GeoSpatial)

    def is_inet(self) -> bool:
        """Return True if an instance of an Inet type."""
        return isinstance(self, INET)

    def is_int16(self) -> bool:
        """Return True if an instance of an Int16 type."""
        return isinstance(self, Int16)

    def is_int32(self) -> bool:
        """Return True if an instance of an Int32 type."""
        return isinstance(self, Int32)

    def is_int64(self) -> bool:
        """Return True if an instance of an Int64 type."""
        return isinstance(self, Int64)

    def is_int8(self) -> bool:
        """Return True if an instance of an Int8 type."""
        return isinstance(self, Int8)

    def is_integer(self) -> bool:
        """Return True if an instance of any Integer type."""
        return isinstance(self, Integer)

    def is_interval(self) -> bool:
        """Return True if an instance of an Interval type."""
        return isinstance(self, Interval)

    def is_json(self) -> bool:
        """Return True if an instance of a JSON type."""
        return isinstance(self, JSON)

    def is_linestring(self) -> bool:
        """Return True if an instance of a LineString type."""
        return isinstance(self, LineString)

    def is_macaddr(self) -> bool:
        """Return True if an instance of a MACADDR type."""
        return isinstance(self, MACADDR)

    def is_map(self) -> bool:
        """Return True if an instance of a Map type."""
        return isinstance(self, Map)

    def is_multilinestring(self) -> bool:
        """Return True if an instance of a MultiLineString type."""
        return isinstance(self, MultiLineString)

    def is_multipoint(self) -> bool:
        """Return True if an instance of a MultiPoint type."""
        return isinstance(self, MultiPoint)

    def is_multipolygon(self) -> bool:
        """Return True if an instance of a MultiPolygon type."""
        return isinstance(self, MultiPolygon)

    def is_nested(self) -> bool:
        """Return true if an instance of any nested (Array/Map/Struct) type."""
        return isinstance(self, (Array, Map, Struct))

    def is_null(self) -> bool:
        """Return true if an instance of a Null type."""
        return isinstance(self, Null)

    def is_numeric(self) -> bool:
        """Return true if an instance of a Numeric type."""
        return isinstance(self, Numeric)

    def is_point(self) -> bool:
        """Return true if an instance of a Point type."""
        return isinstance(self, Point)

    def is_polygon(self) -> bool:
        """Return true if an instance of a Polygon type."""
        return isinstance(self, Polygon)

    def is_primitive(self) -> bool:
        """Return true if an instance of a Primitive type."""
        return isinstance(self, Primitive)

    def is_signed_integer(self) -> bool:
        """Return true if an instance of a SignedInteger type."""
        return isinstance(self, SignedInteger)

    def is_string(self) -> bool:
        """Return true if an instance of a String type."""
        return isinstance(self, String)

    def is_struct(self) -> bool:
        """Return true if an instance of a Struct type."""
        return isinstance(self, Struct)

    def is_temporal(self) -> bool:
        """Return true if an instance of a Temporal type."""
        return isinstance(self, Temporal)

    def is_time(self) -> bool:
        """Return true if an instance of a Time type."""
        return isinstance(self, Time)

    def is_timestamp(self) -> bool:
        """Return true if an instance of a Timestamp type."""
        return isinstance(self, Timestamp)

    def is_uint16(self) -> bool:
        """Return true if an instance of a UInt16 type."""
        return isinstance(self, UInt16)

    def is_uint32(self) -> bool:
        """Return true if an instance of a UInt32 type."""
        return isinstance(self, UInt32)

    def is_uint64(self) -> bool:
        """Return true if an instance of a UInt64 type."""
        return isinstance(self, UInt64)

    def is_uint8(self) -> bool:
        """Return true if an instance of a UInt8 type."""
        return isinstance(self, UInt8)

    def is_unknown(self) -> bool:
        """Return true if an instance of an Unknown type."""
        return isinstance(self, Unknown)

    def is_unsigned_integer(self) -> bool:
        """Return true if an instance of an UnsignedInteger type."""
        return isinstance(self, UnsignedInteger)

    def is_uuid(self) -> bool:
        """Return true if an instance of a UUID type."""
        return isinstance(self, UUID)

    def is_variadic(self) -> bool:
        """Return true if an instance of a Variadic type."""
        return isinstance(self, Variadic)


@public
class Unknown(DataType, Singleton):
    """An unknown type."""

    scalar = "UnknownScalar"
    column = "UnknownColumn"


@public
class Primitive(DataType, Singleton):
    """Values with known size."""


# TODO(kszucs): consider to remove since we don't actually use this information
@public
class Variadic(DataType):
    """Values with unknown size."""


@public
class Parametric(DataType):
    """Types that can be parameterized."""


@public
class Null(Primitive):
    """Null values."""

    scalar = "NullScalar"
    column = "NullColumn"


@public
class Boolean(Primitive):
    """[](`True`) or [](`False`) values."""

    scalar = "BooleanScalar"
    column = "BooleanColumn"


@public
class Bounds(NamedTuple):
    """The lower and upper bound of a fixed-size value."""

    lower: int
    upper: int

    def __contains__(self, value: int) -> bool:
        return self.lower <= value <= self.upper


@public
class Numeric(DataType):
    """Numeric types."""


@public
class Integer(Primitive, Numeric):
    """Integer values."""

    scalar = "IntegerScalar"
    column = "IntegerColumn"

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """Return the number of bytes used to store values of this type."""


@public
class String(Variadic, Singleton):
    """A type representing a string.

    Notes
    -----
    Because of differences in the way different backends handle strings, we
    cannot assume that strings are UTF-8 encoded.
    """

    scalar = "StringScalar"
    column = "StringColumn"


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

    scalar = "BinaryScalar"
    column = "BinaryColumn"


@public
class Temporal(DataType):
    """Data types related to time."""


@public
class Date(Temporal, Primitive):
    """Date values."""

    scalar = "DateScalar"
    column = "DateColumn"


@public
class Time(Temporal, Primitive):
    """Time values."""

    scalar = "TimeScalar"
    column = "TimeColumn"


@public
class Timestamp(Temporal, Parametric):
    """Timestamp values."""

    timezone: Optional[str] = None
    """The timezone of values of this type."""

    # Literal[*range(10)] is only supported from 3.11
    scale: Optional[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] = None
    """The scale of the timestamp if known."""

    scalar = "TimestampScalar"
    column = "TimestampColumn"

    @classmethod
    def from_unit(cls, unit, timezone=None, nullable=True):
        """Return a timestamp type with the given unit and timezone."""
        unit = TimestampUnit(unit)
        if unit == TimestampUnit.SECOND:
            scale = 0
        elif unit == TimestampUnit.MILLISECOND:
            scale = 3
        elif unit == TimestampUnit.MICROSECOND:
            scale = 6
        elif unit == TimestampUnit.NANOSECOND:
            scale = 9
        else:
            raise ValueError(f"Invalid unit {unit}")
        return cls(scale=scale, timezone=timezone, nullable=nullable)

    @property
    def unit(self) -> str:
        """Return the unit of the timestamp."""
        if self.scale is None or self.scale == 0:
            return TimestampUnit.SECOND
        elif 1 <= self.scale <= 3:
            return TimestampUnit.MILLISECOND
        elif 4 <= self.scale <= 6:
            return TimestampUnit.MICROSECOND
        elif 7 <= self.scale <= 9:
            return TimestampUnit.NANOSECOND
        else:
            raise ValueError(f"Invalid scale {self.scale}")

    @property
    def _pretty_piece(self) -> str:
        if self.scale is not None and self.timezone is not None:
            return f"('{self.timezone}', {self.scale:d})"
        elif self.timezone is not None:
            return f"('{self.timezone}')"
        elif self.scale is not None:
            return f"({self.scale:d})"
        else:
            return ""


@public
class SignedInteger(Integer):
    """Signed integer values."""

    @property
    def bounds(self):
        exp = self.nbytes * 8 - 1
        upper = (1 << exp) - 1
        return Bounds(lower=~upper, upper=upper)


@public
class UnsignedInteger(Integer):
    """Unsigned integer values."""

    @property
    def bounds(self):
        exp = self.nbytes * 8
        upper = (1 << exp) - 1
        return Bounds(lower=0, upper=upper)


@public
class Floating(Primitive, Numeric):
    """Floating point values."""

    scalar = "FloatingScalar"
    column = "FloatingColumn"

    @property
    @abstractmethod
    def nbytes(self) -> int:  # pragma: no cover
        """Return the number of bytes used to store values of this type."""


@public
class Int8(SignedInteger):
    """Signed 8-bit integers."""

    nbytes = 1


@public
class Int16(SignedInteger):
    """Signed 16-bit integers."""

    nbytes = 2


@public
class Int32(SignedInteger):
    """Signed 32-bit integers."""

    nbytes = 4


@public
class Int64(SignedInteger):
    """Signed 64-bit integers."""

    nbytes = 8


@public
class UInt8(UnsignedInteger):
    """Unsigned 8-bit integers."""

    nbytes = 1


@public
class UInt16(UnsignedInteger):
    """Unsigned 16-bit integers."""

    nbytes = 2


@public
class UInt32(UnsignedInteger):
    """Unsigned 32-bit integers."""

    nbytes = 4


@public
class UInt64(UnsignedInteger):
    """Unsigned 64-bit integers."""

    nbytes = 8


@public
class Float16(Floating):
    """16-bit floating point numbers."""

    nbytes = 2


@public
class Float32(Floating):
    """32-bit floating point numbers."""

    nbytes = 4


@public
class Float64(Floating):
    """64-bit floating point numbers."""

    nbytes = 8


@public
class Decimal(Numeric, Parametric):
    """Fixed-precision decimal values."""

    precision: Optional[int] = None
    """The number of decimal places values of this type can hold."""

    scale: Optional[int] = None
    """The number of values after the decimal point."""

    scalar = "DecimalScalar"
    column = "DecimalColumn"

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
                raise ValueError("Decimal type precision cannot be negative")
            if not precision:
                raise ValueError("Decimal type precision cannot be zero")
        if scale is not None:
            if not isinstance(scale, numbers.Integral):
                raise TypeError("Decimal type scale must be an integer")
            if scale < 0:
                raise ValueError("Decimal type scale cannot be negative")
            if precision is not None and precision < scale:
                raise ValueError(
                    "Decimal type precision must be greater than or equal to "
                    f"scale. Got precision={precision:d} and scale={scale:d}"
                )
        super().__init__(precision=precision, scale=scale, **kwargs)

    @property
    def _pretty_piece(self) -> str:
        precision = self.precision
        scale = self.scale
        if precision is None and scale is None:
            return ""

        args = [str(precision) if precision is not None else "_"]

        if scale is not None:
            args.append(str(scale))

        return f"({', '.join(args)})"


@public
class Interval(Parametric):
    """Interval values."""

    unit: IntervalUnit
    """The time unit of the interval."""

    scalar = "IntervalScalar"
    column = "IntervalColumn"

    @property
    def resolution(self):
        """The interval unit's name."""
        return self.unit.singular

    @property
    def _pretty_piece(self) -> str:
        return f"('{self.unit.value}')"


@public
class Struct(Parametric, MapSet):
    """Structured values."""

    fields: FrozenDict[str, DataType]

    scalar = "StructScalar"
    column = "StructColumn"

    @classmethod
    def from_tuples(
        cls, pairs: Iterable[tuple[str, str | DataType]], nullable: bool = True
    ) -> Struct:
        """Construct a `Struct` type from pairs.

        Parameters
        ----------
        pairs
            An iterable of pairs of field name and type
        nullable
            Whether the type is nullable

        Returns
        -------
        Struct
            Struct data type instance
        """
        return cls(dict(pairs), nullable=nullable)

    @attribute
    def names(self) -> tuple[str, ...]:
        """Return the names of the struct's fields."""
        return tuple(self.keys())

    @attribute
    def types(self) -> tuple[DataType, ...]:
        """Return the types of the struct's fields."""
        return tuple(self.values())

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __getitem__(self, key: str) -> DataType:
        return self.fields[key]

    def __repr__(self) -> str:
        return f"'{self.name}({list(self.items())}, nullable={self.nullable})"

    @property
    def _pretty_piece(self) -> str:
        pairs = ", ".join(map("{}: {}".format, self.names, self.types))
        return f"<{pairs}>"


T = TypeVar("T", bound=DataType, covariant=True)


@public
class Array(Variadic, Parametric, Generic[T]):
    """Array values."""

    value_type: T

    scalar = "ArrayScalar"
    column = "ArrayColumn"

    @property
    def _pretty_piece(self) -> str:
        return f"<{self.value_type}>"


K = TypeVar("K", bound=DataType, covariant=True)
V = TypeVar("V", bound=DataType, covariant=True)


@public
class Map(Variadic, Parametric, Generic[K, V]):
    """Associative array values."""

    key_type: K
    value_type: V

    scalar = "MapScalar"
    column = "MapColumn"

    @property
    def _pretty_piece(self) -> str:
        return f"<{self.key_type}, {self.value_type}>"


@public
class JSON(Variadic):
    """JSON values."""

    scalar = "JSONScalar"
    column = "JSONColumn"


@public
class GeoSpatial(DataType):
    """Geospatial values."""

    geotype: Optional[Literal["geography", "geometry"]] = None
    """The specific geospatial type."""

    srid: Optional[int] = None
    """The spatial reference identifier."""

    column = "GeoSpatialColumn"
    scalar = "GeoSpatialScalar"

    @property
    def _pretty_piece(self) -> str:
        piece = ""
        if self.geotype is not None:
            piece += f":{self.geotype}"
        if self.srid is not None:
            piece += f";{self.srid}"
        return piece


@public
class Point(GeoSpatial):
    """A point described by two coordinates."""

    scalar = "PointScalar"
    column = "PointColumn"


@public
class LineString(GeoSpatial):
    """A sequence of 2 or more points."""

    scalar = "LineStringScalar"
    column = "LineStringColumn"


@public
class Polygon(GeoSpatial):
    """A set of one or more closed line strings.

    The first line string represents the shape (external ring) and the
    rest represent holes in that shape (internal rings).
    """

    scalar = "PolygonScalar"
    column = "PolygonColumn"


@public
class MultiLineString(GeoSpatial):
    """A set of one or more line strings."""

    scalar = "MultiLineStringScalar"
    column = "MultiLineStringColumn"


@public
class MultiPoint(GeoSpatial):
    """A set of one or more points."""

    scalar = "MultiPointScalar"
    column = "MultiPointColumn"


@public
class MultiPolygon(GeoSpatial):
    """A set of one or more polygons."""

    scalar = "MultiPolygonScalar"
    column = "MultiPolygonColumn"


@public
class UUID(DataType):
    """A 128-bit number used to identify information in computer systems."""

    scalar = "UUIDScalar"
    column = "UUIDColumn"


@public
class MACADDR(DataType):
    """Media Access Control (MAC) address of a network interface."""

    scalar = "MACADDRScalar"
    column = "MACADDRColumn"


@public
class INET(DataType):
    """IP addresses."""

    scalar = "INETScalar"
    column = "INETColumn"


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
# geo spatial data type
geometry = GeoSpatial(geotype="geometry")
geography = GeoSpatial(geotype="geography")
point = Point()
linestring = LineString()
polygon = Polygon()
multilinestring = MultiLineString()
multipoint = MultiPoint()
multipolygon = MultiPolygon()
# json
json = JSON()
# special string based data type
uuid = UUID()
macaddr = MACADDR()
inet = INET()
decimal = Decimal()
unknown = Unknown()

Enum = String


public(
    Any=DataType,
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
    geometry=geometry,
    geography=geography,
    point=point,
    linestring=linestring,
    polygon=polygon,
    multilinestring=multilinestring,
    multipoint=multipoint,
    multipolygon=multipolygon,
    json=json,
    uuid=uuid,
    macaddr=macaddr,
    inet=inet,
    decimal=decimal,
    unknown=unknown,
    Enum=Enum,
    Geography=GeoSpatial,
    Geometry=GeoSpatial,
    Set=Array,
)
