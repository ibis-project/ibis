import pytest
import ibis
import ibis.expr.datatypes as dt


@pytest.mark.parametrize(
    ['arg', 'type'],
    [
        ([1, 2, 3], dt.Array(dt.int8)),
        ([1, 2, 3.0], dt.Array(dt.double)),
        (['a', 'b', 'c'], dt.Array(dt.string))
    ]
)
def test_array_literal(arg, type):
    x = ibis.literal(arg)
    assert x._arg.value == arg
    assert x.type() == type
