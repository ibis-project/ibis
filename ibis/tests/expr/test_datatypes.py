import datetime
import enum
from collections import OrderedDict

import pandas as pd
import parsy
import pytest
import pytz

import ibis
import ibis.expr.datatypes as dt
from ibis.common.exceptions import IbisTypeError


def test_validate_type():
    assert dt.validate_type is dt.dtype


@pytest.mark.parametrize(
    ('spec', 'expected'),
    [
        ('ARRAY<DOUBLE>', dt.Array(dt.double)),
        ('array<array<string>>', dt.Array(dt.Array(dt.string))),
        ('map<string, double>', dt.Map(dt.string, dt.double)),
        (
            'map<int64, array<map<string, int8>>>',
            dt.Map(dt.int64, dt.Array(dt.Map(dt.string, dt.int8))),
        ),
        ('set<uint8>', dt.Set(dt.uint8)),
        ([dt.uint8], dt.Array(dt.uint8)),
        ([dt.float32, dt.float64], dt.Array(dt.float64)),
        ({dt.string}, dt.Set(dt.string)),
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


def test_array_with_string_value_type():
    assert dt.Array('int32') == dt.Array(dt.int32)
    assert dt.Array(dt.Array('array<map<string, double>>')) == (
        dt.Array(dt.Array(dt.Array(dt.Map(dt.string, dt.double))))
    )


def test_map_with_string_value_type():
    assert dt.Map('int32', 'double') == dt.Map(dt.int32, dt.double)
    assert dt.Map('int32', 'array<double>') == dt.Map(dt.int32, dt.Array(dt.double))


def test_map_does_not_allow_non_primitive_keys():
    with pytest.raises(parsy.ParseError):
        dt.dtype('map<array<string>, double>')


def test_token_error():
    with pytest.raises(parsy.ParseError):
        dt.dtype('array<string>>')


def test_empty_complex_type():
    with pytest.raises(parsy.ParseError):
        dt.dtype('map<>')


def test_struct():
    orders = """array<struct<
                    oid: int64,
                    status: string,
                    totalprice: decimal(12, 2),
                    order_date: string,
                    items: array<struct<
                        iid: int64,
                        name: string,
                        price: decimal(12, 2),
                        discount_perc: decimal(12, 2),
                        shipdate: string
                    >>
                >>"""
    expected = dt.Array(
        dt.Struct.from_tuples(
            [
                ('oid', dt.int64),
                ('status', dt.string),
                ('totalprice', dt.Decimal(12, 2)),
                ('order_date', dt.string),
                (
                    'items',
                    dt.Array(
                        dt.Struct.from_tuples(
                            [
                                ('iid', dt.int64),
                                ('name', dt.string),
                                ('price', dt.Decimal(12, 2)),
                                ('discount_perc', dt.Decimal(12, 2)),
                                ('shipdate', dt.string),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    assert dt.dtype(orders) == expected


def test_struct_with_string_types():
    result = dt.Struct.from_tuples(
        [
            ('a', 'map<double, string>'),
            ('b', 'array<map<string, array<int32>>>'),
            ('c', 'array<string>'),
            ('d', 'int8'),
        ]
    )

    assert result == dt.Struct.from_tuples(
        [
            ('a', dt.Map(dt.double, dt.string)),
            ('b', dt.Array(dt.Map(dt.string, dt.Array(dt.int32)))),
            ('c', dt.Array(dt.string)),
            ('d', dt.int8),
        ]
    )


def test_struct_from_dict():
    with pytest.warns(FutureWarning):
        result = dt.Struct.from_dict({'b': 'int64', 'a': dt.float64})

    assert result == dt.Struct({'b': dt.int64, 'a': dt.float64})


def test_struct_names_and_types_legth_must_match():
    with pytest.raises(IbisTypeError), pytest.warns(FutureWarning):
        dt.Struct(names=["a", "b"], types=["int", "str", "float"])

    dtype = dt.Struct({"a": "int", "b": "str"})
    assert isinstance(dtype, dt.Struct)


@pytest.mark.parametrize(
    'case',
    [
        'decimal(',
        'decimal()',
        'decimal(3)',
        'decimal(,)',
        'decimal(3,)',
        'decimal(3,',
    ],
)
def test_decimal_failure(case):
    with pytest.raises(parsy.ParseError):
        dt.dtype(case)


@pytest.mark.parametrize('spec', ['varchar', 'varchar(10)', 'char', 'char(10)'])
def test_char_varchar(spec):
    assert dt.dtype(spec) == dt.string


@pytest.mark.parametrize(
    'spec', ['varchar(', 'varchar)', 'varchar()', 'char(', 'char)', 'char()']
)
def test_char_varchar_invalid(spec):
    with pytest.raises(parsy.ParseError):
        dt.dtype(spec)


@pytest.mark.parametrize(
    ('spec', 'expected'),
    [
        ('boolean', dt.boolean),
        ('int8', dt.int8),
        ('int16', dt.int16),
        ('int32', dt.int32),
        ('int64', dt.int64),
        ('int', dt.int64),
        ('uint8', dt.uint8),
        ('uint16', dt.uint16),
        ('uint32', dt.uint32),
        ('uint64', dt.uint64),
        ('float16', dt.float16),
        ('float32', dt.float32),
        ('float64', dt.float64),
        ('float', dt.float64),
        ('string', dt.string),
        ('binary', dt.binary),
        ('date', dt.date),
        ('time', dt.time),
        ('timestamp', dt.timestamp),
        ('interval', dt.interval),
        ('point', dt.point),
        ('linestring', dt.linestring),
        ('polygon', dt.polygon),
        ('multilinestring', dt.multilinestring),
        ('multipoint', dt.multipoint),
        ('multipolygon', dt.multipolygon),
    ],
)
def test_primitive_from_string(spec, expected):
    assert dt.dtype(spec) == expected


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


def test_literal_mixed_type_fails():
    data = [1, 'a']
    with pytest.raises(TypeError):
        ibis.literal(data)


def test_timestamp_literal_without_tz():
    now_raw = datetime.datetime.utcnow()
    assert now_raw.tzinfo is None
    assert ibis.literal(now_raw).type().timezone is None


def test_timestamp_literal_with_tz():
    now_raw = datetime.datetime.utcnow()
    now_utc = pytz.utc.localize(now_raw)
    assert now_utc.tzinfo == pytz.UTC
    assert ibis.literal(now_utc).type().timezone == str(pytz.UTC)


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


def test_timestamp_with_timezone_parser_single_quote():
    t = dt.dtype("timestamp('US/Eastern')")
    assert isinstance(t, dt.Timestamp)
    assert t.timezone == 'US/Eastern'


def test_timestamp_with_timezone_parser_double_quote():
    t = dt.dtype("timestamp('US/Eastern')")
    assert isinstance(t, dt.Timestamp)
    assert t.timezone == 'US/Eastern'


def test_timestamp_with_timezone_parser_invalid_timezone():
    ts = dt.dtype("timestamp('US/Ea')")
    assert str(ts) == "timestamp('US/Ea')"


@pytest.mark.parametrize(
    'unit',
    [
        'Y',
        'Q',
        'M',
        'W',
        'D',  # date units
        'h',
        'm',
        's',
        'ms',
        'us',
        'ns',  # time units
    ],
)
def test_interval(unit):
    definition = f"interval('{unit}')"
    dt.Interval(unit, dt.int32) == dt.dtype(definition)  # noqa: B015

    definition = f"interval<uint16>('{unit}')"
    dt.Interval(unit, dt.uint16) == dt.dtype(definition)  # noqa: B015

    definition = f"interval<int64>('{unit}')"
    dt.Interval(unit, dt.int64) == dt.dtype(definition)  # noqa: B015


def test_interval_invalid_type():
    with pytest.raises(TypeError):
        dt.Interval('m', dt.float32)

    with pytest.raises(TypeError):
        dt.dtype("interval<float>('s')")


@pytest.mark.parametrize('unit', ['H', 'unsupported'])
def test_interval_invalid_unit(unit):
    definition = f"interval('{unit}')"

    with pytest.raises(ValueError):
        dt.dtype(definition)

    with pytest.raises(ValueError):
        dt.Interval(dt.int32, unit)


@pytest.mark.parametrize(
    'case',
    [
        "timestamp(US/Ea)",
        "timestamp('US/Eastern\")",
        'timestamp("US/Eastern\')',
        "interval(Y)",
        "interval('Y\")",
        'interval("Y\')',
    ],
)
def test_string_argument_parsing_failure_mode(case):
    with pytest.raises(parsy.ParseError):
        dt.dtype(case)


def test_timestamp_with_invalid_timezone():
    ts = dt.Timestamp('Foo/Bar&234')
    assert str(ts) == "timestamp('Foo/Bar&234')"


def test_timestamp_with_timezone_repr():
    ts = dt.Timestamp('UTC')
    assert repr(ts) == "Timestamp(timezone='UTC', scale=None, nullable=True)"


def test_timestamp_with_timezone_str():
    ts = dt.Timestamp('UTC')
    assert str(ts) == "timestamp('UTC')"


def test_time():
    ts = dt.time
    assert str(ts) == "time"


def test_time_valid():
    assert dt.dtype('time').equals(dt.time)


class Foo(enum.Enum):
    a = 1
    b = 2


@pytest.mark.parametrize(
    ('value', 'expected_dtype'),
    [
        (None, dt.null),
        (False, dt.boolean),
        (True, dt.boolean),
        ('foo', dt.string),
        (b'fooblob', dt.binary),
        (datetime.date.today(), dt.date),
        (datetime.datetime.now(), dt.timestamp),
        (datetime.timedelta(days=3), dt.Interval(unit='D')),
        (pd.Timedelta('5 hours'), dt.Interval(unit='h')),
        (pd.Timedelta('7 minutes'), dt.Interval(unit='m')),
        (datetime.timedelta(seconds=9), dt.Interval(unit='s')),
        (pd.Timedelta('11 milliseconds'), dt.Interval(unit='ms')),
        (datetime.timedelta(microseconds=15), dt.Interval(unit='us')),
        (pd.Timedelta('17 nanoseconds'), dt.Interval(unit='ns')),
        # numeric types
        (5, dt.int8),
        (5, dt.int8),
        (127, dt.int8),
        (128, dt.int16),
        (32767, dt.int16),
        (32768, dt.int32),
        (2147483647, dt.int32),
        (2147483648, dt.int64),
        (-5, dt.int8),
        (-128, dt.int8),
        (-129, dt.int16),
        (-32769, dt.int32),
        (-2147483649, dt.int64),
        (1.5, dt.double),
        # parametric types
        (list('abc'), dt.Array(dt.string)),
        (set('abc'), dt.Set(dt.string)),
        ({1, 5, 5, 6}, dt.Set(dt.int8)),
        (frozenset(list('abc')), dt.Set(dt.string)),
        ([1, 2, 3], dt.Array(dt.int8)),
        ([1, 128], dt.Array(dt.int16)),
        ([1, 128, 32768], dt.Array(dt.int32)),
        ([1, 128, 32768, 2147483648], dt.Array(dt.int64)),
        ({'a': 1, 'b': 2, 'c': 3}, dt.Map(dt.string, dt.int8)),
        ({1: 2, 3: 4, 5: 6}, dt.Map(dt.int8, dt.int8)),
        (
            {'a': [1.0, 2.0], 'b': [], 'c': [3.0]},
            dt.Map(dt.string, dt.Array(dt.double)),
        ),
        (
            OrderedDict(
                [
                    ('a', 1),
                    ('b', list('abc')),
                    ('c', OrderedDict([('foo', [1.0, 2.0])])),
                ]
            ),
            dt.Struct.from_tuples(
                [
                    ('a', dt.int8),
                    ('b', dt.Array(dt.string)),
                    (
                        'c',
                        dt.Struct.from_tuples([('foo', dt.Array(dt.double))]),
                    ),
                ]
            ),
        ),
        (Foo.a, dt.Enum()),
    ],
)
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype
    # test literal creation
    value = ibis.literal(value, type=expected_dtype)
    assert value.type() == expected_dtype


@pytest.mark.parametrize(
    ('source', 'target'),
    [
        (dt.string, dt.uuid),
        (dt.uuid, dt.string),
        (dt.null, dt.date),
        (dt.int8, dt.int64),
        (dt.int8, dt.Decimal(12, 2)),
        (dt.int16, dt.uint64),
        (dt.int32, dt.int32),
        (dt.int32, dt.int64),
        (dt.uint32, dt.uint64),
        (dt.uint32, dt.int64),
        (dt.uint32, dt.Decimal(12, 2)),
        (dt.uint32, dt.float32),
        (dt.uint32, dt.float64),
        (dt.uint64, dt.int64),
        (dt.Interval('s', dt.int16), dt.Interval('s', dt.int32)),
    ],
)
def test_implicit_castable(source, target):
    assert dt.castable(source, target)


@pytest.mark.parametrize(
    ('source', 'target'),
    [
        (dt.string, dt.null),
        (dt.int32, dt.int16),
        (dt.int32, dt.uint16),
        (dt.uint64, dt.int16),
        (dt.uint64, dt.uint16),
        (dt.Decimal(12, 2), dt.int32),
        (dt.timestamp, dt.boolean),
        (dt.boolean, dt.interval),
        (dt.Interval('s', dt.int64), dt.Interval('s', dt.int16)),
    ],
)
def test_implicitly_uncastable(source, target):
    assert not dt.castable(source, target)


@pytest.mark.parametrize(
    ('source', 'target', 'value'),
    [(dt.int8, dt.boolean, 0), (dt.int8, dt.boolean, 1)],
)
def test_implicit_castable_values(source, target, value):
    assert dt.castable(source, target, value=value)


@pytest.mark.parametrize(
    ('source', 'target', 'value'),
    [(dt.int8, dt.boolean, 3), (dt.int8, dt.boolean, -1)],
)
def test_implicitly_uncastable_values(source, target, value):
    assert not dt.castable(source, target, value=value)


def test_struct_datatype_subclass_from_tuples():
    class MyStruct(dt.Struct):
        pass

    dtype = MyStruct.from_tuples([('a', 'int64')])
    assert isinstance(dtype, MyStruct)


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
        dt.Set,
        dt.SignedInteger,
        dt.Struct,
        dt.Temporal,
        dt.UnsignedInteger,
        dt.Variadic,
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
