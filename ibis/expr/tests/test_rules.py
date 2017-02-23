import pytest

from ibis.common import IbisTypeError
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.expr import rules


class MyExpr(ir.Expr):
    pass


def test_enum_validator():
    enum = pytest.importorskip('enum')

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


@pytest.mark.parametrize(
    ['options', 'expected_case'],
    [
        (['FOO', 'BAR', 'BAZ'], str.upper),
        (['Foo', 'Bar', 'Baz'], str.upper),  # default is upper
        (['foo', 'bar', 'BAZ'], str.lower),  # majority wins
        (['foo', 'bar', 'Baz'], str.lower),
        (['FOO', 'BAR', 'bAz'], str.upper),
        (['FOO', 'BAR', 'baz'], str.upper),
    ]
)
@pytest.mark.parametrize(
    'option',
    ['foo', 'Foo', 'fOo', 'FOo', 'foO', 'FoO', 'fOO', 'FOO',
     'bar', 'Bar', 'bAr', 'BAr', 'baR', 'BaR', 'bAR', 'BAR',
     'baz', 'Baz', 'bAz', 'BAz', 'baZ', 'BaZ', 'baZ', 'BAZ'],
)
def test_string_options_case_insensitive(options, expected_case, option):
    class MyOp(ops.Node):

        input_type = [
            rules.string_options(options, case_sensitive=False, name='value')
        ]

        def __init__(self, value):
            super(MyOp, self).__init__([value])

        def output_type(self):
            return MyExpr

    op = MyOp(option)
    assert op._validate_args(op.args) == [expected_case(option)]
