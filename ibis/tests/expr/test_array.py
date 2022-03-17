import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops


@pytest.mark.parametrize(
    ['arg', 'typestr', 'type'],
    [
        ([1, 2, 3], None, dt.Array(dt.int8)),
        ([1, 2, 3], 'array<int16>', dt.Array(dt.int16)),
        ([1, 2, 3.0], None, dt.Array(dt.double)),
        (['a', 'b', 'c'], None, dt.Array(dt.string)),
    ],
)
def test_array_literal(arg, typestr, type):
    x = ibis.literal(arg, type=typestr)
    assert x._arg.value == tuple(arg)
    assert x.type() == type


def test_array_length_scalar():
    raw_value = [1, 2, 4]
    value = ibis.literal(raw_value)
    expr = value.length()
    assert isinstance(expr.op(), ops.ArrayLength)
