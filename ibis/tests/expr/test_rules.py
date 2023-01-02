import decimal

import parsy
import pytest
from pytest import param
from toolz import identity

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.common.exceptions import IbisTypeError

table = ibis.table(
    [('int_col', 'int64'), ('string_col', 'string'), ('double_col', 'double')]
)

similar_table = ibis.table(
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
        ('exception', parsy.ParseError),
        ('array<cat>', parsy.ParseError),
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
        (ir.Table, object, IbisTypeError),
        (ir.IntegerValue, 4, IbisTypeError),
    ],
)
def test_invalid_instance_of(klass, value, expected):
    with pytest.raises(expected):
        assert rlz.instance_of(klass, value)


def test_lazy_instance_of():
    rule = rlz.lazy_instance_of("decimal.Decimal")
    assert "decimal.Decimal" in repr(rule)
    d = decimal.Decimal(1)
    assert rule(d) == d
    with pytest.raises(IbisTypeError, match=r"decimal\.Decimal"):
        rule(1)


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
        pytest.param(
            dt.Array(dt.float64),
            [3.4, 5.6],
            ibis.literal([3.4, 5.6]),
        ),
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
    assert result == expected.op()


@pytest.mark.parametrize(
    ('dtype', 'value', 'expected'),
    [
        (dt.uint8, -3, IbisTypeError),
        (dt.int32, {}, IbisTypeError),
        (dt.string, 1, IbisTypeError),
        (dt.Array(dt.float64), ['s'], IbisTypeError),
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


@pytest.mark.parametrize(
    ('validator', 'values', 'expected'),
    [
        param(
            rlz.tuple_of(rlz.integer),
            (3, 2),
            ibis.sequence([3, 2]),
            id="list_int",
        ),
        param(
            rlz.tuple_of(rlz.integer),
            (3, None),
            ibis.sequence([3, ibis.NA]),
            id="list_int_null",
        ),
        param(
            rlz.tuple_of(rlz.string),
            ('a',),
            ibis.sequence(['a']),
            id="list_string_one",
        ),
        param(
            rlz.tuple_of(rlz.string),
            ['a', 'b'],
            ibis.sequence(['a', 'b']),
            id="list_string_two",
        ),
        # TODO(kszucs): currently this is disallowed, probably shouldn't
        # support this case
        # param(
        #     rlz.list_of(rlz.list_of(rlz.string)),
        #     [[], ['a']],
        #     ibis.sequence([ibis.sequence([]), ibis.sequence(['a'])]),
        #     id="list_list_string",
        # ),
        param(
            rlz.tuple_of(rlz.boolean, min_length=2),
            [True, False],
            ibis.sequence([True, False]),
            id="boolean",
        ),
    ],
)
def test_valid_tuple_of(validator, values, expected):
    result = validator(values)
    assert isinstance(result, tuple)


def test_valid_list_of_extra():
    validator = rlz.tuple_of(identity)
    assert validator((3, 2)) == (3, 2)

    validator = rlz.tuple_of(rlz.tuple_of(rlz.string))
    result = validator([[], ['a']])
    assert result[1][0].equals(ibis.literal('a').op())


@pytest.mark.parametrize(
    ('validator', 'values'),
    [
        (rlz.tuple_of(rlz.double, min_length=2), [1]),
        (rlz.tuple_of(rlz.integer), 1.1),
        (rlz.tuple_of(rlz.string), 'asd'),
        (rlz.tuple_of(identity), 3),
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
    assert result.equals(expected.op())


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
    assert result.equals(expected.op())


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
    ('check_table', 'value', 'expected'),
    [
        (table, "int_col", table.int_col),
        (table, table.int_col, table.int_col),
    ],
)
def test_valid_column_from(check_table, value, expected):
    validator = rlz.column_from(rlz.ref("table"))
    this = dict(table=table.op())
    assert validator(value, this=this).equals(expected.op())


@pytest.mark.parametrize(
    ('check_table', 'validator', 'value'),
    [
        (table, rlz.column_from(rlz.ref("not_table")), "int_col"),
        (table, rlz.column_from(rlz.ref("table")), "col_not_in_table"),
        (
            table,
            rlz.column_from(rlz.ref("table")),
            similar_table.int_col,
        ),
    ],
)
def test_invalid_column_from(check_table, validator, value):
    test = dict(table=check_table.op())

    with pytest.raises(IbisTypeError):
        validator(value, this=test)


@pytest.mark.parametrize(
    'table',
    [
        ibis.table([('group', dt.int64), ('value', dt.double)]),
        ibis.table([('group', dt.int64), ('value', dt.double), ('value2', dt.double)]),
    ],
)
def test_table_with_schema(table):
    validator = rlz.table(schema=[('group', dt.int64), ('value', dt.double)])
    assert validator(table) == table.op()


@pytest.mark.parametrize(
    'table', [ibis.table([('group', dt.int64), ('value', dt.timestamp)])]
)
def test_table_with_schema_invalid(table):
    validator = rlz.table(schema=[('group', dt.double), ('value', dt.timestamp)])
    with pytest.raises(IbisTypeError):
        validator(table)


@pytest.mark.parametrize(
    ('validator', 'input'),
    [
        (rlz.tuple_of(rlz.integer), (3, 2)),
        (rlz.instance_of(int), 32),
    ],
)
def test_optional(validator, input):
    expected = validator(input)
    if isinstance(expected, ibis.Expr):
        assert rlz.optional(validator).validate(input).equals(expected)
    else:
        assert rlz.optional(validator).validate(input) == expected
    assert rlz.optional(validator).validate(None) is None


def test_base_table_of_failure_mode():
    class BrokenUseOfBaseTableOf(ops.Node):
        arg = rlz.any
        foo = rlz.function_of(rlz.base_table_of(rlz.ref("arg"), strict=True))

    arg = ibis.literal("abc")

    with pytest.raises(IbisTypeError, match="doesn't have a base table"):
        BrokenUseOfBaseTableOf(arg, identity)
