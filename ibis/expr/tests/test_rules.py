import enum

import pytest
from toolz import identity

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.common.exceptions import IbisTypeError

table = ibis.table(
    [('int_col', 'int64'), ('string_col', 'string'), ('double_col', 'double')]
)


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (dt.int32, dt.int32),
        ('int64', dt.int64),
        ('array<string>', dt.Array(dt.string)),
    ],
)
def test_valid_datatype(value, expected):
    assert rlz.datatype(value) == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('exception', IbisTypeError),
        ('array<cat>', IbisTypeError),
        (int, IbisTypeError),
        ([float], IbisTypeError),
    ],
)
def test_invalid_datatype(value, expected):
    with pytest.raises(expected):
        assert rlz.datatype(value)


@pytest.mark.parametrize(
    ('klass', 'value', 'expected'),
    [(int, 32, 32), (str, 'foo', 'foo'), (dt.Integer, dt.int8, dt.int8)],
)
def test_valid_instance_of(klass, value, expected):
    assert rlz.instance_of(klass, value) == expected


@pytest.mark.parametrize(
    ('klass', 'value', 'expected'),
    [
        (ir.TableExpr, object, IbisTypeError),
        (ir.IntegerValue, 4, IbisTypeError),
    ],
)
def test_invalid_instance_of(klass, value, expected):
    with pytest.raises(expected):
        assert rlz.instance_of(klass, value)


@pytest.mark.parametrize(
    ('dtype', 'value', 'expected'),
    [
        pytest.param(dt.int8, 26, ibis.literal(26)),
        pytest.param(dt.int16, 26, ibis.literal(26)),
        pytest.param(dt.int32, 26, ibis.literal(26)),
        pytest.param(dt.int64, 26, ibis.literal(26)),
        pytest.param(dt.uint8, 26, ibis.literal(26)),
        pytest.param(dt.uint16, 26, ibis.literal(26)),
        pytest.param(dt.uint32, 26, ibis.literal(26)),
        pytest.param(dt.uint64, 26, ibis.literal(26)),
        pytest.param(dt.float32, 26, ibis.literal(26)),
        pytest.param(dt.float64, 26.4, ibis.literal(26.4)),
        pytest.param(dt.double, 26.3, ibis.literal(26.3)),
        pytest.param(dt.string, 'bar', ibis.literal('bar')),
        pytest.param(dt.Array(dt.float), [3.4, 5.6], ibis.literal([3.4, 5.6])),
        pytest.param(
            dt.Map(dt.string, dt.Array(dt.boolean)),
            {'a': [True, False], 'b': [True]},
            ibis.literal({'a': [True, False], 'b': [True]}),
            id='map_literal',
        ),
    ],
)
def test_valid_value(dtype, value, expected):
    result = rlz.value(dtype, value)
    assert result.equals(expected)


@pytest.mark.parametrize(
    ('dtype', 'value', 'expected'),
    [
        (dt.uint8, -3, IbisTypeError),
        (dt.int32, dict(), IbisTypeError),
        (dt.string, 1, IbisTypeError),
        (dt.Array(dt.float), ['s'], IbisTypeError),
        (
            dt.Map(dt.string, dt.Array(dt.boolean)),
            {'a': [True, False], 'b': ['B']},
            IbisTypeError,
        ),
    ],
)
def test_invalid_value(dtype, value, expected):
    with pytest.raises(expected):
        rlz.value(dtype, value)


@pytest.mark.parametrize(
    ('values', 'value', 'expected'),
    [
        (['a', 'b'], 'a', 'a'),
        (('a', 'b'), 'b', 'b'),
        ({'a', 'b', 'c'}, 'c', 'c'),
        ([1, 2, 'f'], 'f', 'f'),
        ({'a': 1, 'b': 2}, 'a', 1),
        ({'a': 1, 'b': 2}, 'b', 2),
    ],
)
def test_valid_isin(values, value, expected):
    assert rlz.isin(values, value) == expected


@pytest.mark.parametrize(
    ('values', 'value', 'expected'),
    [
        (['a', 'b'], 'c', ValueError),
        ({'a', 'b', 'c'}, 'd', ValueError),
        ({'a': 1, 'b': 2}, 'c', ValueError),
    ],
)
def test_invalid_isin(values, value, expected):
    with pytest.raises(expected):
        rlz.isin(values, value)


class Foo(enum.Enum):
    a = 1
    b = 2


class Bar:
    a = 'A'
    b = 'B'


class Baz:
    def __init__(self, a):
        self.a = a


@pytest.mark.parametrize(
    ('obj', 'value', 'expected'),
    [
        (Foo, Foo.a, Foo.a),
        (Foo, 'b', Foo.b),
        (Bar, 'a', 'A'),
        (Bar, 'b', 'B'),
        (Baz(2), 'a', 2),
    ],
)
def test_valid_member_of(obj, value, expected):
    assert rlz.member_of(obj, value) == expected


@pytest.mark.parametrize(
    ('obj', 'value', 'expected'),
    [
        (Foo, 'c', IbisTypeError),
        (Bar, 'c', IbisTypeError),
        (Baz(3), 'b', IbisTypeError),
    ],
)
def test_invalid_member_of(obj, value, expected):
    with pytest.raises(expected):
        rlz.member_of(obj, value)


@pytest.mark.parametrize(
    ('validator', 'values', 'expected'),
    [
        (rlz.list_of(identity), (3, 2), ibis.sequence([3, 2])),
        (rlz.list_of(rlz.integer), (3, 2), ibis.sequence([3, 2])),
        (rlz.list_of(rlz.integer), (3, None), ibis.sequence([3, ibis.NA])),
        (rlz.list_of(rlz.string), ('a',), ibis.sequence(['a'])),
        (rlz.list_of(rlz.string), ['a', 'b'], ibis.sequence(['a', 'b'])),
        pytest.param(
            rlz.list_of(rlz.list_of(rlz.string)),
            [[], ['a']],
            ibis.sequence([[], ['a']]),
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Not yet implemented'
            ),
        ),
        (
            rlz.list_of(rlz.boolean, min_length=2),
            [True, False],
            ibis.sequence([True, False]),
        ),
    ],
)
def test_valid_list_of(validator, values, expected):
    result = validator(values)
    assert result.equals(expected)


@pytest.mark.parametrize(
    ('validator', 'values'),
    [
        (rlz.list_of(rlz.double, min_length=2), [1]),
        (rlz.list_of(rlz.integer), 1.1),
        (rlz.list_of(rlz.string), 'asd'),
        (rlz.list_of(identity), 3),
    ],
)
def test_invalid_list_of(validator, values):
    with pytest.raises(IbisTypeError):
        validator(values)


@pytest.mark.parametrize(
    ('units', 'value', 'expected'),
    [
        ({'H', 'D'}, ibis.interval(days=3), ibis.interval(days=3)),
        (['Y'], ibis.interval(years=3), ibis.interval(years=3)),
    ],
)
def test_valid_interval(units, value, expected):
    result = rlz.interval(value, units=units)
    assert result.equals(expected)


@pytest.mark.parametrize(
    ('units', 'value', 'expected'),
    [
        ({'Y'}, ibis.interval(hours=1), IbisTypeError),
        ({'Y', 'M', 'D'}, ibis.interval(hours=1), IbisTypeError),
        ({'Q', 'W', 'D'}, ibis.interval(seconds=1), IbisTypeError),
    ],
)
def test_invalid_interval(units, value, expected):
    with pytest.raises(expected):
        rlz.interval(value, units=units)


@pytest.mark.parametrize(
    ('validator', 'value', 'expected'),
    [
        (rlz.column(rlz.any), table.int_col, table.int_col),
        (rlz.column(rlz.string), table.string_col, table.string_col),
        (rlz.scalar(rlz.integer), ibis.literal(3), ibis.literal(3)),
        (rlz.scalar(rlz.any), 'caracal', ibis.literal('caracal')),
    ],
)
def test_valid_column_or_scalar(validator, value, expected):
    result = validator(value)
    assert result.equals(expected)


@pytest.mark.parametrize(
    ('validator', 'value', 'expected'),
    [
        (rlz.column(rlz.integer), table.double_col, IbisTypeError),
        (rlz.column(rlz.any), ibis.literal(3), IbisTypeError),
        (rlz.column(rlz.integer), ibis.literal(3), IbisTypeError),
    ],
)
def test_invalid_column_or_scalar(validator, value, expected):
    with pytest.raises(expected):
        validator(value)


@pytest.mark.parametrize(
    'table',
    [
        ibis.table([('group', dt.int64), ('value', dt.double)]),
        ibis.table(
            [('group', dt.int64), ('value', dt.double), ('value2', dt.double)]
        ),
    ],
)
def test_table_with_schema(table):
    validator = rlz.table([('group', dt.int64), ('value', dt.double)])
    assert validator(table) == table


@pytest.mark.parametrize(
    'table', [ibis.table([('group', dt.int64), ('value', dt.timestamp)])]
)
def test_table_with_schema_invalid(table):
    validator = rlz.table([('group', dt.double), ('value', dt.timestamp)])
    with pytest.raises(IbisTypeError):
        validator(table)


def test_shape_like_with_no_arguments():
    with pytest.raises(ValueError) as e:
        rlz.shape_like([])
    assert str(e.value) == 'Must pass at least one expression'


@pytest.mark.parametrize(
    ('rule', 'input'),
    [
        (rlz.array_of(rlz.integer), [1, 2, 3]),
        (rlz.array_of(rlz.integer), []),
        (rlz.array_of(rlz.double), [1, 2]),
        (rlz.array_of(rlz.string), ['a', 'b']),
        (rlz.array_of(rlz.array_of(rlz.string)), [['a'], [], [], ['a', 'b']]),
    ],
)
def test_array_of(rule, input):
    assert isinstance(rule(input).type(), dt.Array)


@pytest.mark.parametrize(
    ('rule', 'input'),
    [
        (rlz.array_of(rlz.array_of(rlz.string)), [1, 2]),
        (rlz.array_of(rlz.string), [1, 2.0]),
        (rlz.array_of(rlz.array_of(rlz.integer)), [2, 2.0]),
    ],
)
def test_array_of_invalid_input(rule, input):
    with pytest.raises(IbisTypeError):
        rule(input)
