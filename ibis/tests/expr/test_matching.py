from matchpy import Wildcard

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

_ = Wildcard.dot()
x = Wildcard.dot('x')
y = Wildcard.dot('y')

args = Wildcard.plus('args')


def test_match_syntax_sugar():
    result = ops.Literal.pattern(_, y) << ops.Literal(1, dt.int8)
    assert result == {'y': dt.int8}
    result = (x, y) << ops.Literal(1, dt.int8)
    assert result == {'x': 1, 'y': dt.int8}
    result = {'dtype': x, 'value': y} << ops.Literal(1, dt.int8)
    assert result == {'y': 1, 'x': dt.int8}

    result = ops.Literal(3, dt.int8) >> ops.Literal.pattern(y, x)
    assert result == {'y': 3, 'x': dt.int8}
    result = ops.Literal(2, dt.int16) >> (x, y)
    assert result == {'x': 2, 'y': dt.int16}
    result = ops.Literal(2, dt.int16) >> {'dtype': x, 'value': y}
    assert result == {'y': 2, 'x': dt.int16}


def test_substitute_syntax_sugar():
    result = ops.Literal.pattern(x, y) << {'x': 1.0, 'y': dt.float64}
    assert result == ops.Literal(1.0, dtype=dt.float64)

    result = {'x': 2.0, 'y': "float32"} >> ops.Literal.pattern(x, y)
    assert result == ops.Literal(2.0, dtype=dt.float32)


def test_match_list():
    assert ops.List.pattern(1, 2, _) << ops.List(1, 2, 3) == {}
    assert (1, 2, _) << ops.List(1, 2, 3) == {}

    assert ops.List.pattern(x, 2, _) << ops.List(1, 2, 3) == {'x': 1}
    assert (x, 2, _) << ops.List(1, 2, 3) == {'x': 1}
    assert (x, 2, y) << ops.List(1, 2, 3) == {'x': 1, 'y': 3}

    assert (x, args) << ops.List(1, 2, 3, 4) == {'x': 1, 'args': (2, 3, 4)}
    assert (x, args, y) << ops.List(1, 2, 3, 4) == {
        'x': 1,
        'y': 4,
        'args': (2, 3),
    }


def test_substitute_list():
    assert {} >> ops.List.pattern(1, 2, x) == ops.List.pattern(1, 2, x)
    assert {'x': 3} >> ops.List.pattern(1, 2, x) == ops.List(1, 2, 3)

    result = {'x': 1, 'args': (2, 3, 4)} >> ops.List.pattern(args, 3, 4, x)
    assert result == ops.List(2, 3, 4, 3, 4, 1)
