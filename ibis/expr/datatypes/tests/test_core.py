from __future__ import annotations

import datetime  # noqa: TCH003
import decimal  # noqa: TCH003
import sys
import uuid  # noqa: TCH003
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import pytest

import ibis.expr.datatypes as dt
from ibis.common.annotations import ValidationError
from ibis.common.patterns import As, Attrs, NoMatch, Pattern
from ibis.common.temporal import TimestampUnit, TimeUnit


def test_validate_type():
    assert dt.validate_type is dt.dtype


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("ARRAY<DOUBLE>", dt.Array(dt.double)),
        ("array<array<string>>", dt.Array(dt.Array(dt.string))),
        ("map<string, double>", dt.Map(dt.string, dt.double)),
        (
            "map<int64, array<map<string, int8>>>",
            dt.Map(dt.int64, dt.Array(dt.Map(dt.string, dt.int8))),
        ),
    ]
    + [
        (f"{cls.__name__.lower()}{suffix}", expected)
        for cls in [
            dt.Point,
            dt.LineString,
            dt.Polygon,
            dt.MultiLineString,
            dt.MultiPoint,
            dt.MultiPolygon,
        ]
        for suffix, expected in [
            ("", cls()),
            (";4326", cls(srid=4326)),
            (";4326:geometry", cls(geotype="geometry", srid=4326)),
            (";4326:geography", cls(geotype="geography", srid=4326)),
        ]
    ],
)
def test_dtype(spec, expected):
    assert dt.dtype(spec) == expected


@pytest.mark.parametrize(
    ("klass", "expected"),
    [
        (dt.Int16, dt.int16),
        (dt.Int32, dt.int32),
        (dt.Int64, dt.int64),
        (dt.UInt8, dt.uint8),
        (dt.UInt16, dt.uint16),
        (dt.UInt32, dt.uint32),
        (dt.UInt64, dt.uint64),
        (dt.Float32, dt.float32),
        (dt.Float64, dt.float64),
        (dt.String, dt.string),
        (dt.Binary, dt.binary),
        (dt.Boolean, dt.boolean),
        (dt.Date, dt.date),
        (dt.Time, dt.time),
    ],
)
def test_dtype_from_classes(klass, expected):
    assert dt.dtype(klass) == expected


@pytest.mark.parametrize(
    ("klass", "lower", "upper"),
    [
        (dt.UInt8, 0, 2**8 - 1),
        (dt.UInt16, 0, 2**16 - 1),
        (dt.UInt32, 0, 2**32 - 1),
        (dt.UInt64, 0, 2**64 - 1),
        (dt.Int8, -(2**7), 2**7 - 1),
        (dt.Int16, -(2**15), 2**15 - 1),
        (dt.Int32, -(2**31), 2**31 - 1),
        (dt.Int64, -(2**63), 2**63 - 1),
    ],
)
def test_signed_unsigned_bounds(klass, lower, upper):
    bounds = dt.dtype(klass).bounds
    assert bounds.lower == lower
    assert bounds.upper == upper


class FooStruct:
    a: dt.int16
    b: dt.int32
    c: dt.int64
    d: dt.uint8
    e: dt.uint16
    f: dt.uint32
    g: dt.uint64
    h: dt.float32
    i: dt.float64
    j: dt.string
    k: dt.binary
    l: dt.boolean  # noqa: E741
    m: dt.date
    n: dt.time
    o: dt.timestamp
    oa: dt.Timestamp("UTC")
    ob: dt.Timestamp("UTC", 6)
    pa: dt.Interval("s")
    q: dt.decimal
    qa: dt.Decimal(12, 2)
    r: dt.Array(dt.int16)
    s: dt.Map(dt.string, dt.int16)


foo_struct = dt.Struct(
    {
        "a": dt.int16,
        "b": dt.int32,
        "c": dt.int64,
        "d": dt.uint8,
        "e": dt.uint16,
        "f": dt.uint32,
        "g": dt.uint64,
        "h": dt.float32,
        "i": dt.float64,
        "j": dt.string,
        "k": dt.binary,
        "l": dt.boolean,
        "m": dt.date,
        "n": dt.time,
        "o": dt.timestamp,
        "oa": dt.Timestamp("UTC"),
        "ob": dt.Timestamp("UTC", 6),
        "pa": dt.Interval("s"),
        "q": dt.decimal,
        "qa": dt.Decimal(12, 2),
        "r": dt.Array(dt.int16),
        "s": dt.Map(dt.string, dt.int16),
    }
)


class BarStruct:
    a: dt.Int16
    b: dt.Int32
    c: dt.Int64
    d: dt.UInt8
    e: dt.UInt16
    f: dt.UInt32
    g: dt.UInt64
    h: dt.Float32
    i: dt.Float64
    j: dt.String
    k: dt.Binary
    l: dt.Boolean  # noqa: E741
    m: dt.Date
    n: dt.Time
    r: dt.Array[dt.Int16]
    s: dt.Map[dt.String, dt.Int16]


bar_struct = dt.Struct(
    {
        "a": dt.int16,
        "b": dt.int32,
        "c": dt.int64,
        "d": dt.uint8,
        "e": dt.uint16,
        "f": dt.uint32,
        "g": dt.uint64,
        "h": dt.float32,
        "i": dt.float64,
        "j": dt.string,
        "k": dt.binary,
        "l": dt.boolean,
        "m": dt.date,
        "n": dt.time,
        "r": dt.Array(dt.int16),
        "s": dt.Map(dt.string, dt.int16),
    }
)


class MyInt(int):
    pass


class MyFloat(float):
    pass


class MyStr(str):
    pass


class MyBytes(bytes):
    pass


class MyList(list):
    pass


class MyTuple(list):
    pass


class MyDict(dict):
    pass


class MyStruct:
    a: str
    b: int
    c: float


class PyStruct:
    a: int
    b: float
    c: str
    ca: MyStr
    d: bytes
    da: MyBytes
    e: bool
    f: datetime.date
    g: datetime.time
    h: datetime.datetime
    i: datetime.timedelta
    j: decimal.Decimal
    k: list[int]
    l: dict[str, int]  # noqa: E741
    n: tuple[str]
    o: uuid.UUID
    p: None
    q: MyStruct


class PyStruct2:
    ka: list[int]
    kb: MyList[int]
    la: dict[str, int]
    lb: MyDict[str, int]
    na: tuple[str]
    nb: MyTuple[str]


py_struct = dt.Struct(
    {
        "a": dt.int64,
        "b": dt.float64,
        "c": dt.string,
        "ca": dt.string,
        "d": dt.binary,
        "da": dt.binary,
        "e": dt.boolean,
        "f": dt.date,
        "g": dt.time,
        "h": dt.timestamp,
        "i": dt.Interval("us"),
        "j": dt.decimal,
        "k": dt.Array(dt.int64),
        "l": dt.Map(dt.string, dt.int64),
        "n": dt.Array(dt.string),
        "o": dt.UUID,
        "p": dt.null,
        "q": dt.Struct(
            {
                "a": dt.string,
                "b": dt.int64,
                "c": dt.float64,
            }
        ),
    }
)
py_struct_2 = dt.Struct(
    {
        "ka": dt.Array(dt.int64),
        "kb": dt.Array(dt.int64),
        "la": dt.Map(dt.string, dt.int64),
        "lb": dt.Map(dt.string, dt.int64),
        "na": dt.Array(dt.string),
        "nb": dt.Array(dt.string),
    }
)


class FooNamedTuple(NamedTuple):
    a: str
    b: int
    c: float


@dataclass
class FooDataClass:
    a: str
    b: int
    c: float = 0.1


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        (dt.Array[dt.Null], dt.Array(dt.Null())),
        (dt.Map[dt.Null, dt.Null], dt.Map(dt.Null(), dt.Null())),
        (FooStruct, foo_struct),
        (BarStruct, bar_struct),
        (PyStruct, py_struct),
        (FooNamedTuple, dt.Struct({"a": dt.string, "b": dt.int64, "c": dt.float64})),
        (FooDataClass, dt.Struct({"a": dt.string, "b": dt.int64, "c": dt.float64})),
    ],
)
def test_dtype_from_typehints(hint, expected):
    assert dt.dtype(hint) == expected


@pytest.mark.parametrize(("hint", "expected"), [(PyStruct2, py_struct_2)])
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
def test_dtype_from_newer_typehints(hint, expected):
    assert dt.dtype(hint) == expected


def test_dtype_from_invalid_python_value():
    msg = "Cannot construct an ibis datatype from python value `1.0`"
    with pytest.raises(TypeError, match=msg):
        dt.dtype(1.0)


def test_dtype_from_invalid_python_type():
    class Something:
        pass

    msg = "Cannot construct an ibis datatype from python type `<class '.*Something'>`"
    with pytest.raises(TypeError, match=msg):
        dt.dtype(Something)


def test_struct_subclass_from_tuples():
    class MyStruct(dt.Struct):
        pass

    dtype = MyStruct.from_tuples([("a", "int64")])
    assert isinstance(dtype, MyStruct)


def test_struct_mapping_api():
    s = dt.Struct(
        {
            "a": dt.Map(dt.float64, dt.string),
            "b": dt.Array(dt.Map(dt.string, dt.Array(dt.int32))),
            "c": dt.Array(dt.string),
            "d": dt.int8,
        }
    )

    assert s["a"] == dt.Map(dt.double, dt.string)
    assert s["b"] == dt.Array(dt.Map(dt.string, dt.Array(dt.int32)))
    assert s["c"] == dt.Array(dt.string)
    assert s["d"] == dt.int8

    assert "a" in s
    assert "e" not in s
    assert len(s) == 4
    assert tuple(s) == s.names
    assert tuple(s.keys()) == s.names
    assert tuple(s.values()) == s.types
    assert tuple(s.items()) == tuple(zip(s.names, s.types))

    s1 = s.copy()
    s2 = dt.Struct(
        {
            "a": dt.Map(dt.float64, dt.string),
            "b": dt.Array(dt.Map(dt.string, dt.Array(dt.int32))),
            "c": dt.Array(dt.string),
        }
    )
    assert s == s1
    assert s != s2

    # doesn't support item assignment
    with pytest.raises(TypeError):
        s["e"] = dt.int8


def test_struct_set_operations():
    a = dt.Struct({"a": dt.string, "b": dt.int64, "c": dt.float64})
    b = dt.Struct({"a": dt.string, "c": dt.float64, "d": dt.boolean, "e": dt.date})
    c = dt.Struct({"i": dt.int64, "j": dt.float64, "k": dt.string})
    d = dt.Struct({"i": dt.int64, "j": dt.float64, "k": dt.string, "l": dt.boolean})

    assert a & b == dt.Struct({"a": dt.string, "c": dt.float64})
    assert a | b == dt.Struct(
        {"a": dt.string, "b": dt.int64, "c": dt.float64, "d": dt.boolean, "e": dt.date}
    )
    assert a - b == dt.Struct({"b": dt.int64})
    assert b - a == dt.Struct({"d": dt.boolean, "e": dt.date})
    assert a ^ b == dt.Struct({"b": dt.int64, "d": dt.boolean, "e": dt.date})

    assert not a.isdisjoint(b)
    assert a.isdisjoint(c)

    assert a <= a
    assert a >= a
    assert not a < a
    assert not a > a
    assert not a <= b
    assert not a >= b
    assert not a >= c
    assert not a <= c
    assert c <= d
    assert c < d
    assert d >= c
    assert d > c


def test_singleton_null():
    assert dt.null is dt.Null()


def test_singleton_boolean():
    assert dt.Boolean() == dt.boolean
    assert dt.Boolean() is dt.boolean
    assert dt.Boolean() is dt.Boolean()
    assert dt.Boolean(nullable=True) is dt.boolean
    assert dt.Boolean(nullable=False) is not dt.boolean
    assert dt.Boolean(nullable=False) is dt.Boolean(nullable=False)
    assert dt.Boolean(nullable=True) is dt.Boolean(nullable=True)
    assert dt.Boolean(nullable=True) is not dt.Boolean(nullable=False)


def test_singleton_primitive():
    assert dt.Int64() is dt.int64
    assert dt.Int64(nullable=False) is not dt.int64
    assert dt.Int64(nullable=False) is dt.Int64(nullable=False)


def test_array_type_not_equals():
    left = dt.Array(dt.string)
    right = dt.Array(dt.int32)

    assert not left.equals(right)
    assert left != right
    assert not (left == right)  # noqa: SIM201


def test_array_type_equals():
    left = dt.Array(dt.string)
    right = dt.Array(dt.string)

    assert left.equals(right)
    assert left == right
    assert not (left != right)  # noqa: SIM202


def test_interval_invalid_value_type():
    with pytest.raises(ValidationError):
        dt.Interval("m", dt.float32)


@pytest.mark.parametrize("unit", ["H", "unsupported"])
def test_interval_invalid_unit(unit):
    with pytest.raises(ValidationError):
        dt.Interval(dt.int32, unit)


def test_timestamp_with_invalid_timezone():
    ts = dt.Timestamp("Foo/Bar&234")
    assert str(ts) == "timestamp('Foo/Bar&234')"


def test_timestamp_with_timezone_repr():
    ts = dt.Timestamp("UTC")
    assert repr(ts) == "Timestamp(timezone='UTC', scale=None, nullable=True)"


def test_timestamp_with_timezone_str():
    ts = dt.Timestamp("UTC")
    assert str(ts) == "timestamp('UTC')"


def test_time_str():
    assert str(dt.time) == "time"


def test_parse_null():
    assert dt.parse("null") == dt.null


@pytest.mark.parametrize("scale", range(10))
@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
def test_timestamp_with_scale(scale, tz):
    assert dt.parse(f"timestamp({tz!r}, {scale:d})") == dt.Timestamp(
        timezone=tz, scale=scale
    )


@pytest.mark.parametrize("scale", range(10))
def test_timestamp_with_scale_no_tz(scale):
    assert dt.parse(f"timestamp({scale:d})") == dt.Timestamp(scale=scale)


def test_timestamp_unit():
    assert dt.Timestamp().unit == TimestampUnit.SECOND
    assert dt.Timestamp(scale=0).unit == TimestampUnit.SECOND
    for scale in range(1, 3):
        assert dt.Timestamp(scale=scale).unit == TimestampUnit.MILLISECOND
    for scale in range(4, 7):
        assert dt.Timestamp(scale=scale).unit == TimestampUnit.MICROSECOND
    for scale in range(7, 10):
        assert dt.Timestamp(scale=scale).unit == TimestampUnit.NANOSECOND


def test_timestamp_from_unit():
    assert dt.Timestamp.from_unit("s") == dt.Timestamp(scale=0)
    assert dt.Timestamp.from_unit("ms", timezone="UTC") == dt.Timestamp(
        scale=3, timezone="UTC"
    )
    assert dt.Timestamp.from_unit("us", nullable=True) == dt.Timestamp(
        scale=6, nullable=True
    )
    assert dt.Timestamp.from_unit("ns", timezone="UTC", nullable=False) == dt.Timestamp(
        scale=9, timezone="UTC", nullable=False
    )


def get_leaf_classes(op):
    for child_class in op.__subclasses__():
        yield child_class
        yield from get_leaf_classes(child_class)


@pytest.mark.parametrize(
    "dtype_class",
    set(get_leaf_classes(dt.DataType))
    - {
        # these require special case tests
        dt.Array,
        dt.Enum,
        dt.Floating,
        dt.GeoSpatial,
        dt.Integer,
        dt.Map,
        dt.Numeric,
        dt.Primitive,
        dt.SignedInteger,
        dt.Struct,
        dt.Temporal,
        dt.UnsignedInteger,
        dt.Variadic,
        dt.Parametric,
        dt.Interval,
    },
)
def test_is_methods(dtype_class):
    name = dtype_class.__name__.lower()
    dtype = getattr(dt, name)
    is_dtype = getattr(dtype, f"is_{name}")()
    assert is_dtype is True


def test_is_array():
    assert dt.Array(dt.string).is_array()
    assert not dt.string.is_array()


def test_is_floating():
    assert dt.float64.is_floating()


def test_is_geospatial():
    assert dt.geometry.is_geospatial()


def test_is_integer():
    assert dt.int32.is_integer()


def test_is_map():
    assert dt.Map(dt.int8, dt.Array(dt.string)).is_map()


def test_is_numeric():
    assert dt.int64.is_numeric()
    assert dt.float32.is_numeric()
    assert dt.decimal.is_numeric()
    assert not dt.string.is_numeric()


def test_is_primitive():
    assert dt.bool.is_primitive()
    assert dt.uint8.is_primitive()
    assert not dt.decimal.is_primitive()


def test_is_signed_integer():
    assert dt.int8.is_signed_integer()
    assert not dt.uint8.is_signed_integer()


def test_is_struct():
    assert dt.Struct({"a": dt.string}).is_struct()


def test_is_unsigned_integer():
    assert dt.uint8.is_unsigned_integer()
    assert not dt.int8.is_unsigned_integer()


def test_is_variadic():
    assert dt.string.is_variadic()
    assert not dt.int8.is_variadic()


def test_is_temporal():
    assert dt.time.is_temporal()
    assert dt.date.is_temporal()
    assert dt.timestamp.is_temporal()
    assert not dt.Array(dt.Map(dt.string, dt.string)).is_temporal()


def test_set_is_an_alias_of_array():
    assert dt.Set is dt.Array


def test_type_coercion():
    p = Pattern.from_typehint(dt.DataType)
    assert p.match(dt.int8, {}) == dt.int8
    assert p.match("int8", {}) == dt.int8
    assert p.match(dt.string, {}) == dt.string
    assert p.match("string", {}) == dt.string
    assert p.match(3, {}) is NoMatch

    p = Pattern.from_typehint(dt.Primitive)
    assert p.match(dt.int8, {}) == dt.int8
    assert p.match("int8", {}) == dt.int8
    assert p.match(dt.boolean, {}) == dt.boolean
    assert p.match("boolean", {}) == dt.boolean
    assert p.match(dt.Array(dt.int8), {}) is NoMatch
    assert p.match("array<int8>", {}) is NoMatch

    p = Pattern.from_typehint(dt.Integer)
    assert p.match(dt.int8, {}) == dt.int8
    assert p.match("int8", {}) == dt.int8
    assert p.match(dt.uint8, {}) == dt.uint8
    assert p.match("uint8", {}) == dt.uint8
    assert p.match(dt.boolean, {}) is NoMatch
    assert p.match("boolean", {}) is NoMatch

    p = Pattern.from_typehint(dt.Array[dt.Integer])
    assert p.match(dt.Array(dt.int8), {}) == dt.Array(dt.int8)
    assert p.match("array<int8>", {}) == dt.Array(dt.int8)
    assert p.match(dt.Array(dt.uint8), {}) == dt.Array(dt.uint8)
    assert p.match("array<uint8>", {}) == dt.Array(dt.uint8)
    assert p.match(dt.Array(dt.boolean), {}) is NoMatch
    assert p.match("array<boolean>", {}) is NoMatch

    p = Pattern.from_typehint(dt.Map[dt.String, dt.Integer])
    assert p.match(dt.Map(dt.string, dt.int8), {}) == dt.Map(dt.string, dt.int8)
    assert p.match("map<string, int8>", {}) == dt.Map(dt.string, dt.int8)
    assert p.match(dt.Map(dt.string, dt.uint8), {}) == dt.Map(dt.string, dt.uint8)
    assert p.match("map<string, uint8>", {}) == dt.Map(dt.string, dt.uint8)
    assert p.match(dt.Map(dt.string, dt.boolean), {}) is NoMatch
    assert p.match("map<string, boolean>", {}) is NoMatch

    p = Pattern.from_typehint(Annotated[dt.Interval, Attrs(unit=As(TimeUnit))])
    assert p.match(dt.Interval("s"), {}) == dt.Interval("s")
    assert p.match(dt.Interval("ns"), {}) == dt.Interval("ns")
