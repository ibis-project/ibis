import pytest
import datetime
import numpy as np
from collections import OrderedDict

import ibis
from ibis import IbisError
from ibis.expr import datatypes as dt
from ibis.expr.rules import highest_precedence_type
import ibis.expr.api as api
import ibis.expr.types as types
import ibis.expr.rules as rules


def test_validate_type():
    assert dt.validate_type is dt.dtype


def test_array():
    assert dt.dtype('ARRAY<DOUBLE>') == dt.Array(dt.double)


def test_nested_array():
    assert dt.dtype('array<array<string>>') == dt.Array(dt.Array(dt.string))


def test_array_with_string_value_type():
    assert dt.Array('int32') == dt.Array(dt.int32)
    assert dt.Array(dt.Array('array<map<string, double>>')) == (
        dt.Array(dt.Array(dt.Array(dt.Map(dt.string, dt.double))))
    )


def test_map():
    assert dt.dtype('map<string, double>') == dt.Map(dt.string, dt.double)


def test_nested_map():
    expected = dt.Map(dt.int64, dt.Array(dt.Map(dt.string, dt.int8)))
    assert dt.dtype('map<int64, array<map<string, int8>>>') == expected


def test_map_with_string_value_type():
    assert dt.Map('int32', 'double') == dt.Map(dt.int32, dt.double)
    assert dt.Map('int32', 'array<double>') == \
        dt.Map(dt.int32, dt.Array(dt.double))


def test_map_does_not_allow_non_primitive_keys():
    with pytest.raises(SyntaxError):
        dt.dtype('map<array<string>, double>')


def test_token_error():
    with pytest.raises(SyntaxError):
        dt.dtype('array<string>>')


def test_empty_complex_type():
    with pytest.raises(SyntaxError):
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
    expected = dt.Array(dt.Struct.from_tuples([
        ('oid', dt.int64),
        ('status', dt.string),
        ('totalprice', dt.Decimal(12, 2)),
        ('order_date', dt.string),
        (
            'items',
            dt.Array(dt.Struct.from_tuples([
                ('iid', dt.int64),
                ('name', dt.string),
                ('price', dt.Decimal(12, 2)),
                ('discount_perc', dt.Decimal(12, 2)),
                ('shipdate', dt.string),
            ]))
        )
    ]))

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


@pytest.mark.parametrize('case', [
    'decimal(',
    'decimal()',
    'decimal(3)',
    'decimal(,)',
    'decimal(3,)',
    'decimal(3,',
])
def test_decimal_failure(case):
    with pytest.raises(SyntaxError):
        dt.dtype(case)


@pytest.mark.parametrize('spec', [
    'varchar',
    'varchar(10)',
    'char',
    'char(10)'
])
def test_char_varchar(spec):
    assert dt.dtype(spec) == dt.string


@pytest.mark.parametrize('spec', [
    'varchar(',
    'varchar)',
    'varchar()',
    'char(',
    'char)',
    'char()'
])
def test_char_varchar_invalid(spec):
    with pytest.raises(SyntaxError):
        dt.dtype(spec)


@pytest.mark.parametrize(('spec', 'expected'), [
    ('any', dt.any),
    ('null', dt.null),
    ('boolean', dt.boolean),
    ('int8', dt.int8),
    ('int16', dt.int16),
    ('int32', dt.int32),
    ('int64', dt.int64),
    ('uint8', dt.uint8),
    ('uint16', dt.uint16),
    ('uint32', dt.uint32),
    ('uint64', dt.uint64),
    ('float16', dt.float16),
    ('float32', dt.float32),
    ('float64', dt.float64),
    ('float', dt.float),
    ('halffloat', dt.float16),
    ('double', dt.double),
    ('string', dt.string),
    ('binary', dt.binary),
    ('date', dt.date),
    ('time', dt.time),
    ('timestamp', dt.timestamp),
    ('interval', dt.interval)
])
def test_primitive(spec, expected):
    assert dt.dtype(spec) == expected


@pytest.mark.parametrize(('numpy_dtype', 'ibis_dtype'), [
    (np.bool_, dt.boolean),
    (np.int8, dt.int8),
    (np.int16, dt.int16),
    (np.int32, dt.int32),
    (np.int64, dt.int64),
    (np.uint8, dt.uint8),
    (np.uint16, dt.uint16),
    (np.uint32, dt.uint32),
    (np.uint64, dt.uint64),
    (np.float16, dt.float16),
    (np.float32, dt.float32),
    (np.float64, dt.float64),
    (np.double, dt.double),
    (np.str_, dt.string),
    (np.datetime64, dt.timestamp),
    (np.timedelta64, dt.interval)
])
def test_numpy_dtype(numpy_dtype, ibis_dtype):
    assert dt.dtype(np.dtype(numpy_dtype)) == ibis_dtype


def test_precedence_with_no_arguments():
    with pytest.raises(ValueError) as e:
        highest_precedence_type([])
    assert str(e.value) == 'Must pass at least one expression'


def test_rule_instance_of():
    class MyOperation(types.Node):

        input_type = [rules.instance_of(types.IntegerValue)]

    MyOperation([api.literal(5)])

    with pytest.raises(IbisError):
        MyOperation([api.literal('string')])


def test_literal_mixed_type_fails():
    data = [1, 'a']
    with pytest.raises(TypeError):
        ibis.literal(data)


def test_array_type_not_equals():
    left = dt.Array(dt.string)
    right = dt.Array(dt.int32)

    assert not left.equals(right)
    assert left != right
    assert not (left == right)


def test_array_type_equals():
    left = dt.Array(dt.string)
    right = dt.Array(dt.string)

    assert left.equals(right)
    assert left == right
    assert not (left != right)


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


@pytest.mark.parametrize('unit', [
    'Y', 'Q', 'M', 'w',  'd',  # date units
    'h',  'm',  's',  'ms', 'us', 'ns'  # time units
])
def test_interval(unit):
    definition = "interval('{}')".format(unit)
    dt.Interval(unit, dt.int32) == dt.dtype(definition)

    definition = "interval<uint16>('{}')".format(unit)
    dt.Interval(unit, dt.uint16) == dt.dtype(definition)

    definition = "interval<int64>('{}')".format(unit)
    dt.Interval(unit, dt.int64) == dt.dtype(definition)


def test_interval_invalid_type():
    with pytest.raises(TypeError):
        dt.Interval('m', dt.float32)

    with pytest.raises(TypeError):
        dt.dtype("interval<float>('s')")


@pytest.mark.parametrize('unit', [
    'H', 'unsupported'
])
def test_interval_unvalid_unit(unit):
    definition = "interval('{}')".format(unit)

    with pytest.raises(ValueError):
        dt.dtype(definition)

    with pytest.raises(ValueError):
        dt.Interval(dt.int32, unit)


@pytest.mark.parametrize('case', [
    "timestamp(US/Ea)",
    "timestamp('US/Eastern\")",
    'timestamp("US/Eastern\')',
    "interval(Y)",
    "interval('Y\")",
    'interval("Y\')',
])
def test_string_argument_parsing_failure_mode(case):
    with pytest.raises(SyntaxError):
        dt.dtype(case)


def test_timestamp_with_invalid_timezone():
    ts = dt.Timestamp('Foo/Bar&234')
    assert str(ts) == "timestamp('Foo/Bar&234')"


def test_timestamp_with_timezone_repr():
    ts = dt.Timestamp('UTC')
    assert repr(ts) == "Timestamp(timezone='UTC', nullable=True)"


def test_timestamp_with_timezone_str():
    ts = dt.Timestamp('UTC')
    assert str(ts) == "timestamp('UTC')"


def test_time():
    ts = dt.time
    assert str(ts) == "time"


def test_time_valid():
    assert dt.dtype('time').equals(dt.time)


@pytest.mark.parametrize(('value', 'expected_dtype'), [
    (None, dt.null),
    (False, dt.boolean),
    (True, dt.boolean),
    ('foo', dt.string),

    (datetime.date.today(), dt.date),
    (datetime.datetime.now(), dt.timestamp),
    (datetime.timedelta(days=3), dt.interval),

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

    # numpy types
    (np.int8(5), dt.int8),
    (np.int16(-1), dt.int16),
    (np.int32(2), dt.int32),
    (np.int64(-5), dt.int64),
    (np.uint8(5), dt.uint8),
    (np.uint16(50), dt.uint16),
    (np.uint32(500), dt.uint32),
    (np.uint64(5000), dt.uint64),
    (np.float32(5.5), dt.float32),
    (np.float32(5.5), dt.float),
    (np.float64(5.55), dt.float64),
    (np.float64(5.55), dt.double),
    (np.bool_(True), dt.boolean),
    (np.bool_(False), dt.boolean),
    (np.arange(5, dtype='int32'), dt.Array(dt.int32)),

    # parametric types
    (list('abc'), dt.Array(dt.string)),
    ([1, 2, 3], dt.Array(dt.int8)),
    ([1, 128], dt.Array(dt.int16)),
    ([1, 128, 32768], dt.Array(dt.int32)),
    ([1, 128, 32768, 2147483648], dt.Array(dt.int64)),
    ({'a': 1, 'b': 2, 'c': 3}, dt.Map(dt.string, dt.int8)),
    ({1: 2, 3: 4, 5: 6}, dt.Map(dt.int8, dt.int8)),
    (
        {'a': [1.0, 2.0], 'b': [], 'c': [3.0]},
        dt.Map(dt.string, dt.Array(dt.double))
    ),
    (
        OrderedDict([
            ('a', 1),
            ('b', list('abc')),
            ('c', OrderedDict([('foo', [1.0, 2.0])]))
        ]),
        dt.Struct.from_tuples([
            ('a', dt.int8),
            ('b', dt.Array(dt.string)),
            ('c', dt.Struct.from_tuples([
                ('foo', dt.Array(dt.double))
            ]))
        ])
    )
])
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


@pytest.mark.parametrize(('source', 'target'), [
    (dt.any, dt.string),
    (dt.null, dt.date),
    (dt.null, dt.any),
    (dt.int8, dt.int64),
    (dt.int8, dt.Decimal(12, 2)),
    (dt.int32, dt.int32),
    (dt.int32, dt.int64),
    (dt.uint32, dt.uint64),
    (dt.uint32, dt.Decimal(12, 2)),
    (dt.uint32, dt.float32),
    (dt.uint32, dt.float64),
    (dt.Interval('s', dt.int16), dt.Interval('s', dt.int32)),
])
def test_implicit_castable(source, target):
    assert dt.castable(source, target)


@pytest.mark.parametrize(('source', 'target'), [
    (dt.string, dt.null),
    (dt.int32, dt.int16),
    (dt.Decimal(12, 2), dt.int32),
    (dt.timestamp, dt.boolean),
    (dt.boolean, dt.interval),
    (dt.Interval('s', dt.int64), dt.Interval('s', dt.int16)),
])
def test_implicitly_uncastable(source, target):
    assert not dt.castable(source, target)


@pytest.mark.parametrize(('source', 'target', 'value'), [
    (dt.int8, dt.boolean, 0),
    (dt.int8, dt.boolean, 1),
])
def test_implicit_castable_values(source, target, value):
    assert dt.castable(source, target, value=value)


@pytest.mark.parametrize(('source', 'target', 'value'), [
    (dt.int8, dt.boolean, 3),
    (dt.int8, dt.boolean, -1),
])
def test_implicitly_uncastable_values(source, target, value):
    assert not dt.castable(source, target, value=value)
