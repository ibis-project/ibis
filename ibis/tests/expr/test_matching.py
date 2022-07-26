from matchpy import Wildcard

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

_ = Wildcard.dot()
x = Wildcard.dot('x')
y = Wildcard.dot('y')

args = Wildcard.plus('args')

one = ops.Literal(1, dt.int8)
two = ops.Literal(2, dt.int8)
three = ops.Literal(3, dt.int8)
four = ops.Literal(4, dt.int8)


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
    assert (
        ops.NodeList.pattern(one, two, _) << ops.NodeList(one, two, three)
        == {}
    )
    assert (one, two, _) << ops.NodeList(one, two, three) == {}

    assert ops.NodeList.pattern(x, two, _) << ops.NodeList(
        one, two, three
    ) == {'x': one}
    assert (x, two, _) << ops.NodeList(one, two, three) == {'x': one}
    assert (x, two, y) << ops.NodeList(one, two, three) == {
        'x': one,
        'y': three,
    }

    assert (x, args) << ops.NodeList(one, two, three, four) == {
        'x': one,
        'args': (two, three, four),
    }
    assert (x, args, y) << ops.NodeList(one, two, three, four) == {
        'x': one,
        'y': four,
        'args': (two, three),
    }


def test_substitute_list():
    assert {} >> ops.NodeList.pattern(one, two, x) == ops.NodeList.pattern(
        one, two, x
    )
    assert {'x': three} >> ops.NodeList.pattern(one, two, x) == ops.NodeList(
        one, two, three
    )

    result = {'x': one, 'args': (two, three, four)} >> ops.NodeList.pattern(
        args, three, four, x
    )
    assert result == ops.NodeList(two, three, four, three, four, one)
