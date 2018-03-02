import six
import enum
import ibis
import pytest

from toolz import identity
from ibis.common import IbisTypeError

import ibis.expr.types as ir
import ibis.expr.rules as rlz
import ibis.expr.datatypes as dt


table = ibis.table([
    ('int_col', 'int64'),
    ('string_col', 'string'),
    ('double_col', 'double'),
])


@pytest.mark.parametrize(('value', 'expected'), [
    (dt.int32, dt.int32),
    ('int64', dt.int64),
    ('array<string>', dt.Array(dt.string)),
])
def test_valid_datatype(value, expected):
    assert rlz.datatype(value) == expected


@pytest.mark.parametrize(('value', 'expected'), [
    ('exception', IbisTypeError),
    ('array<cat>', IbisTypeError),
    (int, IbisTypeError),
    ([float], IbisTypeError)
])
def test_invalid_datatype(value, expected):
    with pytest.raises(expected):
        assert rlz.datatype(value)


@pytest.mark.parametrize(('klass', 'value', 'expected'), [
    (int, 32, 32),
    (six.string_types, 'foo', 'foo'),
    (dt.Integer, dt.int8, dt.int8),
])
def test_valid_instance_of(klass, value, expected):
    assert rlz.instance_of(klass, value) == expected


@pytest.mark.parametrize(('klass', 'value', 'expected'), [
    (ir.TableExpr, object, IbisTypeError),
    (ir.IntegerValue, 4, IbisTypeError)
])
def test_invalid_instance_of(klass, value, expected):
    with pytest.raises(expected):
        assert rlz.instance_of(klass, value)


@pytest.mark.parametrize(('dtype', 'value', 'expected'), [
    (dt.int8, 26, ibis.literal(26)),
    (dt.int16, 26, ibis.literal(26)),
    (dt.int32, 26, ibis.literal(26)),
    (dt.int64, 26, ibis.literal(26)),
    (dt.uint8, 26, ibis.literal(26)),
    (dt.uint16, 26, ibis.literal(26)),
    (dt.uint32, 26, ibis.literal(26)),
    (dt.uint64, 26, ibis.literal(26)),
    (dt.float32, 26, ibis.literal(26)),
    (dt.float64, 26.4, ibis.literal(26.4)),
    (dt.double, 26.3, ibis.literal(26.3)),
    (dt.string, 'bar', ibis.literal('bar')),
    (dt.Array(dt.float), [3.4, 5.6], ibis.literal([3.4, 5.6])),
    (dt.Map(dt.string, dt.Array(dt.boolean)),
     {'a': [True, False], 'b': [True]},
     ibis.literal({'a': [True, False], 'b': [True]})),
], ids=lambda x: str(x.value if isinstance(x, ir.ValueExpr) else x))
def test_valid_value(dtype, value, expected):
    result = rlz.value(dtype, value)
    assert result.equals(expected)


@pytest.mark.parametrize(('dtype', 'value', 'expected'), [
    (dt.uint8, -3, IbisTypeError),
    (dt.int32, dict(), IbisTypeError),
    (dt.string, 1, IbisTypeError),
    (dt.Array(dt.float), ['s'], IbisTypeError),
    (dt.Map(dt.string, dt.Array(dt.boolean)),
     {'a': [True, False], 'b': ['B']},
     IbisTypeError)
])
def test_invalid_value(dtype, value, expected):
    with pytest.raises(expected):
        rlz.value(dtype, value)


@pytest.mark.parametrize(('validator', 'value', 'expected'), [
    (rlz.optional(identity), None, None),
    (rlz.optional(identity), 'three', 'three'),
    (rlz.optional(identity, default=1), None, 1),
    (rlz.optional(identity, default=lambda: 8), 'cat', 'cat'),
    (rlz.optional(identity, default=lambda: 8), None, 8),
    (rlz.optional(rlz.instance_of(int), default=11), None, 11),
    (rlz.optional(rlz.instance_of(int)), None, None),
    (rlz.optional(rlz.instance_of(int)), 18, 18),
    (rlz.optional(rlz.instance_of(str)), 'caracal', 'caracal'),
])
def test_valid_optional(validator, value, expected):
    assert validator(value) == expected


@pytest.mark.parametrize(('validator', 'value', 'expected'), [
    (rlz.optional(rlz.instance_of(int), default=''), None, IbisTypeError),
    (rlz.optional(rlz.instance_of(int)), 'lynx', IbisTypeError),
])
def test_invalid_optional(validator, value, expected):
    with pytest.raises(expected):
        validator(value)


@pytest.mark.parametrize(('values', 'value', 'expected'), [
    (['a', 'b'], 'a', 'a'),
    (('a', 'b'), 'b', 'b'),
    ({'a', 'b', 'c'}, 'c', 'c'),
    ([1, 2, 'f'], 'f', 'f'),
    ({'a': 1, 'b': 2}, 'a', 1),
    ({'a': 1, 'b': 2}, 'b', 2),
])
def test_valid_isin(values, value, expected):
    assert rlz.isin(values, value) == expected


@pytest.mark.parametrize(('values', 'value', 'expected'), [
    (['a', 'b'], 'c', ValueError),
    ({'a', 'b', 'c'}, 'd', ValueError),
    ({'a': 1, 'b': 2}, 'c', ValueError),
])
def test_invalid_isin(values, value, expected):
    with pytest.raises(expected):
        rlz.isin(values, value)


class Foo(enum.Enum):
    a = 1
    b = 2


class Bar(object):
    a = 'A'
    b = 'B'


class Baz(object):

    def __init__(self, a):
        self.a = a


@pytest.mark.parametrize(('obj', 'value', 'expected'), [
    (Foo, Foo.a, Foo.a),
    (Foo, 'b', Foo.b),
    (Bar, 'a', 'A'),
    (Bar, 'b', 'B'),
    (Baz(2), 'a', 2),
])
def test_valid_member_of(obj, value, expected):
    assert rlz.member_of(obj, value) == expected


@pytest.mark.parametrize(('obj', 'value', 'expected'), [
    (Foo, 'c', IbisTypeError),
    (Bar, 'c', IbisTypeError),
    (Baz(3), 'b', IbisTypeError)
])
def test_invalid_member_of(obj, value, expected):
    with pytest.raises(expected):
        rlz.member_of(obj, value)


@pytest.mark.parametrize(('validator', 'values', 'expected'), [
    (rlz.list_of(identity), 3, ibis.sequence([3])),
    (rlz.list_of(identity), (3, 2), ibis.sequence([3, 2])),
    (rlz.list_of(rlz.integer), (3, 2), ibis.sequence([3, 2])),
    (rlz.list_of(rlz.integer), (3, None), ibis.sequence([3, ibis.NA])),
    (rlz.list_of(rlz.string), 'asd', ibis.sequence(['asd'])),
    (rlz.list_of(rlz.boolean, min_length=2), [True, False],
     ibis.sequence([True, False]))
])
def test_valid_list_of(validator, values, expected):
    result = validator(values)
    assert result.equals(expected)


@pytest.mark.parametrize(('validator', 'values', 'expected'), [
    (rlz.list_of(rlz.double, min_length=2), [1], IbisTypeError),
    (rlz.list_of(rlz.integer), 1.1, IbisTypeError),
])
def test_invalid_list_of(validator, values, expected):
    with pytest.raises(expected):
        validator(values)


@pytest.mark.parametrize(('units', 'value', 'expected'), [
    ({'H', 'D'}, ibis.interval(days=3), ibis.interval(days=3)),
    (['Y'], ibis.interval(years=3), ibis.interval(years=3)),
])
def test_valid_interval(units, value, expected):
    result = rlz.interval(value, units=units)
    assert result.equals(expected)


@pytest.mark.parametrize(('units', 'value', 'expected'), [
    ({'Y'}, ibis.interval(hours=1), IbisTypeError),
    ({'Y', 'M', 'D'}, ibis.interval(hours=1), IbisTypeError),
    ({'Q', 'W', 'D'}, ibis.interval(seconds=1), IbisTypeError)
])
def test_invalid_interval(units, value, expected):
    with pytest.raises(expected):
        rlz.interval(value, units=units)


@pytest.mark.parametrize(('validator', 'value', 'expected'), [
    (rlz.column(rlz.any), table.int_col, table.int_col),
    (rlz.column(rlz.string), table.string_col, table.string_col),
    (rlz.scalar(rlz.integer), ibis.literal(3), ibis.literal(3)),
    (rlz.scalar(rlz.any), 'caracal', ibis.literal('caracal'))
])
def test_valid_column_or_scalar(validator, value, expected):
    result = validator(value)
    assert result.equals(expected)


@pytest.mark.parametrize(('validator', 'value', 'expected'), [
    (rlz.column(rlz.integer), table.double_col, IbisTypeError),
    (rlz.column(rlz.any), ibis.literal(3), IbisTypeError),
    (rlz.column(rlz.integer), ibis.literal(3), IbisTypeError),
])
def test_invalid_column_or_scalar(validator, value, expected):
    with pytest.raises(expected):
        validator(value)


def test_custom_list_of_as_value_expr():

    class MyList(list):
        pass

    class MyEnum(enum.Enum):
        A = 1
        B = 2

    class MyEnum2(enum.Enum):
        A = 1
        B = '2'

    def custom_as_value_expr(o):
        if o and all(isinstance(el.value, six.integer_types) for el in o):
            return MyList(o)
        return o

    class MyOp(ops.ValueOp):

        input_type = [
            rules.list_of(
                rules.enum(MyEnum),
                name='one',
                as_value_expr=custom_as_value_expr
            ),
            rules.list_of(
                rules.enum(MyEnum2),
                name='two',
                as_value_expr=custom_as_value_expr
            ),
        ]

    result = MyOp([MyEnum.A, MyEnum.B], [])
    assert isinstance(result.one, MyList)
    assert result.one == [MyEnum.A, MyEnum.B]
    assert result.two == []

    result = MyOp([MyEnum.A, MyEnum.B], [MyEnum2.B])
    assert isinstance(result.one, MyList)
    assert not isinstance(result.two, MyList)
    assert result.one == [MyEnum.A, MyEnum.B]
    assert result.two == [MyEnum2.B]


def check_op_input(Op, schema, raises):
    schema = ibis.Schema.from_tuples(schema)
    table = ibis.table(schema)
    if not raises:
        assert Op(table).table.equals(table)
    else:
        with pytest.raises(IbisTypeError):
            Op(table)


@pytest.mark.parametrize(
    'schema',
    [[('group', dt.int64),
      ('value', dt.double)],

     [('group', dt.int64),
      ('value', dt.double),
      ('value2', dt.double)]])
def test_table_with_schema(schema):
    class MyOp(ops.ValueOp):
        input_type = [
            rules.table(
                name='table',
                schema=rules.table.with_column_subset(
                    rules.column(name='group', value_type=rules.number),
                    rules.column(name='value', value_type=rules.number)
                ))]
        output_type = rules.type_of_arg(0)

    schema = ibis.Schema.from_tuples(schema)
    table = ibis.table(schema)
    MyOp(table)


@pytest.mark.parametrize(
    'schema',
    [[('group', dt.int64),
      ('value', dt.timestamp)]])
def test_table_with_schema_invalid(schema):
    class MyOp(ops.ValueOp):
        input_type = [
            rules.table(
                name='table',
                schema=rules.table.with_column_subset(
                    rules.column(name='group', value_type=rules.number),
                    rules.column(name='value', value_type=rules.number)
                ))]
        output_type = rules.type_of_arg(0)

    schema = ibis.Schema.from_tuples(schema)
    table = ibis.table(schema)
    with pytest.raises(IbisTypeError):
        MyOp(table)


@pytest.mark.parametrize(
    'schema',
    [[('group', dt.int64),
      ('value', dt.double)],

     [('group', dt.int64),
      ('value', dt.double),
      ('value2', dt.double)],

     [('group', dt.int64),
      ('value', dt.double),
      ('value3', dt.timestamp)],

     [('group', dt.int64),
      ('value', dt.double),
      ('value2', dt.double),
      ('value3', dt.double)],

     [('group', dt.int64),
      ('value', dt.double),
      ('value2', dt.double),
      ('value3', dt.timestamp)]])
def test_table_with_schema_optional(schema):
    class MyOp(ops.ValueOp):
        input_type = [
            rules.table(
                name='table',
                schema=rules.table.with_column_subset(
                    rules.column(name='group', value_type=rules.number),
                    rules.column(name='value', value_type=rules.number),
                    rules.column(name='value2', value_type=rules.number,
                                 optional=True)))]
        output_type = rules.type_of_arg(0)

    schema = ibis.Schema.from_tuples(schema)
    table = ibis.table(schema)
    MyOp(table)


@pytest.mark.parametrize(
    'schema',
    [[('group', dt.int64),
      ('value', dt.double),
      ('value2', dt.timestamp)]])
def test_table_with_schema_optional_invalid(schema):
    class MyOp(ops.ValueOp):
        input_type = [
            rules.table(
                name='table',
                schema=rules.table.with_column_subset(
                    rules.column(name='group', value_type=rules.number),
                    rules.column(name='value', value_type=rules.number),
                    rules.column(name='value2', value_type=rules.number,
                                 optional=True)))]
        output_type = rules.type_of_arg(0)

    schema = ibis.Schema.from_tuples(schema)
    table = ibis.table(schema)
    with pytest.raises(IbisTypeError):
        MyOp(table)


def test_table_not_a_table():
    class MyOp(ops.ValueOp):
        input_type = [rules.table(name='table')]
        output_type = rules.type_of_arg(0)

    with pytest.raises(IbisTypeError):
        MyOp(123)


def test_table_invalid_schema_no_name():
    with pytest.raises(ValueError):
        class MyOp(ops.ValueOp):
            input_type = [rules.table(
                name='table',
                schema=rules.table.with_column_subset(
                    rules.column(value_type=rules.number)))]
            output_type = rules.type_of_arg(0)


def test_table_invalid_schema_wrong_class():
    with pytest.raises(ValueError):
        class MyOp(ops.ValueOp):
            input_type = [rules.table(
                name='table',
                schema='wrong class')]
            output_type = rules.type_of_arg(0)


def test_table_invalid_column_subset():
    with pytest.raises(ValueError):
        class MyOp(ops.ValueOp):
            input_type = [rules.table(
                name='table',
                schema=rules.table.with_column_subset('not a rule'))]
            output_type = rules.type_of_arg(0)


def test_table_custom_validator():
    class MyOp(ops.ValueOp):
        input_type = [rules.table(
            name='table',
            validator=lambda x: 1 / 0)]
        output_type = rules.type_of_arg(0)

    schema = ibis.Schema.from_tuples([('group', dt.int64)])
    table = ibis.table(schema)

    with pytest.raises(ZeroDivisionError):
        MyOp(table)


def test_shapeof_with_no_arguments():
    with pytest.raises(ValueError) as e:
        rlz.shapeof([])
    assert str(e.value) == 'Must pass at least one expression'
