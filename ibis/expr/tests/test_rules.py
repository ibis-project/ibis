import enum
import pytest

import six

import ibis
from ibis.common import IbisTypeError
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
from ibis.expr import rules
import ibis.expr.rlz as rlz


def test_validator():
    # TODO
    pass


class MyExpr(ir.Expr):
    pass


def test_enum_validator():
    class Foo(enum.Enum):
        a = 1
        b = 2

    class Bar(enum.Enum):
        a = 1
        b = 2

    class MyOp(ops.Node):

        input_type = [rules.enum(Foo, name='value')]

        def __init__(self, value):
            super(MyOp, self).__init__([value])

        def output_type(self):
            return MyExpr

    assert MyOp(2) is not None
    assert MyOp(Foo.b) is not None

    with pytest.raises(IbisTypeError):
        MyOp(3)

    with pytest.raises(IbisTypeError):
        MyOp(Bar.a)

    op = MyOp(Foo.a)
    assert op._validate_args(op.args) == [Foo.a]

    op = MyOp(2)
    assert op._validate_args(op.args) == [Foo.b]


def test_duplicate_enum():
    enum = pytest.importorskip('enum')

    class Dup(enum.Enum):
        a = 1
        b = 1
        c = 2

    class MyOp(ops.Node):

        input_type = [rules.enum(Dup, name='value')]

        def __init__(self, value):
            super(MyOp, self).__init__([value])

        def output_type(self):
            return MyExpr

    with pytest.raises(IbisTypeError):
        MyOp(1)

    assert MyOp(2) is not None


# case sensitivity feature nowhere used
# @pytest.mark.parametrize(
#     ['options', 'expected_case'],
#     [
#         (['FOO', 'BAR', 'BAZ'], str.upper),
#         (['Foo', 'Bar', 'Baz'], str.upper),  # default is upper
#         (['foo', 'bar', 'BAZ'], str.lower),  # majority wins
#         (['foo', 'bar', 'Baz'], str.lower),
#         (['FOO', 'BAR', 'bAz'], str.upper),
#         (['FOO', 'BAR', 'baz'], str.upper),
#     ]
# )
# @pytest.mark.parametrize(
#     'option',
#     ['foo', 'Foo', 'fOo', 'FOo', 'foO', 'FoO', 'fOO', 'FOO',
#      'bar', 'Bar', 'bAr', 'BAr', 'baR', 'BaR', 'bAR', 'BAR',
#      'baz', 'Baz', 'bAz', 'BAz', 'baZ', 'BaZ', 'baZ', 'BAZ'],
# )
# def test_string_options_case_insensitive(options, expected_case, option):
#     class MyOp(ops.Node):
#         value = rlz.string_options(options, case_sensitive=False)
#         # input_type = [
#         #     rules.string_options(options, case_sensitive=False, name='value')
# #        ]

#         def output_type(self):
#             return MyExpr

#     op = MyOp(option)
#     print(op.value)
#     assert op._validate_args(op.args) == [expected_case(option)]


def test_argument_docstring():
    doc = 'A wonderful integer'

    class MyExpr(ir.Expr):
        pass

    class MyOp(ops.ValueOp):

        input_type = [rules.integer(name='foo', doc=doc)]

        def output_type(self):
            return MyExpr

    op = MyOp(1)
    assert type(op).foo.__doc__ == doc


def test_scalar_value_type():

    class MyOp(ops.ValueOp):
        arg = rlz.scalar(rlz.numeric)
        output_type = rules.double

    with pytest.raises(IbisTypeError):
        MyOp('a')

    assert MyOp(1).args[0].equals(ibis.literal(1))
    assert MyOp(1.42).args[0].equals(ibis.literal(1.42))


def test_array_rule():

    class MyOp(ops.ValueOp):
        value = rlz.value(dt.Array(dt.double))
        output_type = rules.type_of_arg(0)

    raw_value = [1.0, 2.0, 3.0]
    op = MyOp(raw_value)
    result = op.value
    expected = ibis.literal(raw_value)
    assert result.equals(expected)


# def test_scalar_default_arg():
#     class MyOp(ops.ValueOp):

#         input_type = [
#             rules.scalar(
#                 value_type=dt.boolean,
#                 optional=True,
#                 default=False,
#                 name='value'
#             )
#         ]
#         output_type = rules.type_of_arg(0)

#     op = MyOp()
#     assert op.value.equals(ibis.literal(False))

#     op = MyOp(True)
#     assert op.value.equals(ibis.literal(True))


def test_rule_instance_of():
    class MyOperation(ops.Node):
        arg = rlz.instanceof(ir.IntegerValue)

    MyOperation(ir.literal(5))


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

    # with pytest.raises(IbisTypeError):
    #     MyOperation(ir.literal('string'))
