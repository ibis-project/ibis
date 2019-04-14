import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


@pytest.mark.parametrize(
    ['arg', 'type'],
    [
        ([1, 2, 3], dt.Array(dt.int8)),
        ([1, 2, 3.0], dt.Array(dt.double)),
        (['a', 'b', 'c'], dt.Array(dt.string)),
    ],
)
def test_array_literal(arg, type):
    x = ibis.literal(arg)
    assert x._arg.value == arg
    assert x.type() == type


def test_array_length_scalar():
    raw_value = [1, 2, 4]
    value = ibis.literal(raw_value)
    expr = value.length()
    assert isinstance(expr.op(), ops.ArrayLength)
