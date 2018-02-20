import enum
import pytest
import six
import ibis
import enum
from contextlib import contextmanager
from ibis.common import IbisTypeError
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
from ibis.expr import rules
import ibis.expr.rlz as rlz
from toolz import identity


def mayraise(error):
    """Wrapper around pytest.raises to support None."""
    if type(error) is type and issubclass(error, Exception):
        return pytest.raises(error)
    else:
        @contextmanager
        def not_raises():
            try:
                yield
            except Exception as e:
                raise e
        return not_raises()


table = ibis.table([
    ('int_col', 'int64'),
    ('string_col', 'string'),
    ('double_col', 'double'),
])


@pytest.mark.parametrize(('value', 'expected'), [
    (dt.int32, dt.int32),
    ('int64', dt.int64),
    ('array<string>', dt.Array(dt.string)),
    ('exception', IbisTypeError),
    ('array<cat>', IbisTypeError),
    (int, IbisTypeError),
    ([float], IbisTypeError)
])
def test_datatype(value, expected):
    with mayraise(expected):
        assert rlz.datatype(value) == expected


@pytest.mark.parametrize(('klass', 'value', 'expected'), [
    (int, 32, 32),
    (ir.TableExpr, object, IbisTypeError),
    (six.string_types, 'foo', 'foo'),
    (dt.Integer, dt.int8, dt.int8),
    (ir.IntegerValue, 4, IbisTypeError)
])
def test_instanceof(klass, value, expected):
    with mayraise(expected):
        assert rlz.instanceof(klass, value) == expected


@pytest.mark.parametrize(('dtype', 'value', 'expected'), [
    (dt.int32, 26, ibis.literal(26)),
    (dt.int32, dict(), IbisTypeError),
    (dt.string, 'bar', ibis.literal('bar')),
    (dt.string, 1, IbisTypeError),
    (dt.Array(dt.float), [3.4, 5.6], ibis.literal([3.4, 5.6])),
    (dt.Array(dt.float), ['s'], IbisTypeError),  # TODO fails because of incorrect subtype cecking
    (dt.Map(dt.string, dt.Array(dt.boolean)),
     {'a': [True, False], 'b': [True]},
     ibis.literal({'a': [True, False], 'b': [True]})),
    (dt.Map(dt.string, dt.Array(dt.boolean)),
     {'a': [True, False], 'b': ['B']},
     IbisTypeError)
])
def test_value(dtype, value, expected):
    with mayraise(expected):
        result = rlz.value(dtype, value)
        assert result.equals(expected)


@pytest.mark.parametrize(('validator', 'value', 'expected'), [
    (rlz.optional(identity), None, None),
    (rlz.optional(identity), 'three', 'three'),
    (rlz.optional(identity, default=1), None, 1),
    (rlz.optional(identity, default=lambda: 8), 'cat', 'cat'),
    (rlz.optional(identity, default=lambda: 8), None, 8),
    (rlz.optional(rlz.instanceof(int), default=''), None, IbisTypeError),
    (rlz.optional(rlz.instanceof(int), default=11), None, 11),
    (rlz.optional(rlz.instanceof(int)), None, None),
    (rlz.optional(rlz.instanceof(int)), 18, 18),
    (rlz.optional(rlz.instanceof(int)), 'lynx', IbisTypeError),
    (rlz.optional(rlz.instanceof(str)), 'caracal', 'caracal'),
])
def test_optional(validator, value, expected):
    with mayraise(expected):
        assert validator(value) == expected


@pytest.mark.parametrize(('values', 'value', 'expected'), [
    (['a', 'b'], 'a', 'a'),
    (('a', 'b'), 'b', 'b'),
    ({'a', 'b', 'c'}, 'c', 'c'),
    (['a', 'b'], 'c', IbisTypeError),
    ({'a', 'b', 'c'}, 'd', IbisTypeError),
    ([1, 2, 'f'], 'f', 'f'),
    ({'a': 1, 'b': 2}, 'a', 1),
    ({'a': 1, 'b': 2}, 'b', 2),
    ({'a': 1, 'b': 2}, 'c', IbisTypeError),
])
def test_isin(values, value, expected):
    with mayraise(expected):
        assert rlz.isin(values, value) == expected


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
    (Foo, 'c', IbisTypeError),
    (Bar, 'c', IbisTypeError),
    (Bar, 'a', 'A'),
    (Bar, 'b', 'B'),
    (Baz(2), 'a', 2),
    (Baz(3), 'b', IbisTypeError)
])
def test_memberof(obj, value, expected):
    with mayraise(expected):
        assert rlz.memberof(obj, value) == expected


@pytest.mark.parametrize(('validator', 'values', 'expected'), [
    (rlz.listof(identity), 3, IbisTypeError),
    (rlz.listof(identity), (3, 2), ir.sequence([3, 2])),
    (rlz.listof(rlz.integer), (3, 2), ir.sequence([3, 2])),
    (rlz.listof(rlz.integer), (3, None), IbisTypeError),
    (rlz.listof(rlz.string), 'asd', IbisTypeError),
    (rlz.listof(rlz.double, min_length=2), [1], IbisTypeError),
    (rlz.listof(rlz.boolean, min_length=2), [True, False],
     ir.sequence([True, False]))
])
def test_listof(validator, values, expected):
    with mayraise(expected):
        result = validator(values)
        assert result.equals(expected)


@pytest.mark.parametrize(('units', 'value', 'expected'), [
    ({'H', 'D'}, ibis.interval(days=3), ibis.interval(days=3)),
    (['Y'], ibis.interval(years=3), ibis.interval(years=3)),
    ({'Y'}, ibis.interval(hours=1), IbisTypeError)
])
def test_interval(units, value, expected):
    with mayraise(expected):
        result = rlz.interval(value, units=units)
        assert result.equals(expected)


@pytest.mark.parametrize(('validator', 'value', 'expected'), [
    (rlz.column(rlz.any), table.int_col, table.int_col),
    (rlz.column(rlz.string), table.string_col, table.string_col),
    (rlz.column(rlz.integer), table.double_col, IbisTypeError),
    (rlz.column(rlz.any), ibis.literal(3), IbisTypeError),
    (rlz.column(rlz.integer), ibis.literal(3), IbisTypeError),
    (rlz.scalar(rlz.integer), ibis.literal(3), ibis.literal(3)),
    (rlz.scalar(rlz.any), 'caracal', ibis.literal('caracal'))
])
def test_column_or_scalar(validator, value, expected):
    with mayraise(expected):
        result = validator(value)
        assert result.equals(expected)


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



