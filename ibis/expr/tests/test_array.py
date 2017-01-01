import pytest
import ibis
import ibis.expr.datatypes as dt


def test_array_literal():
    x = ibis.literal([1, 2, 3])
    assert x._arg.value == [1, 2, 3]
    assert x.type() == dt.Array(dt.int8)


def test_array_literal_mixed():
    x = ibis.literal([1, 2, 3.0])
    assert x._arg.value == [1, 2, 3.0]
    assert x.type() == dt.Array(dt.double)
