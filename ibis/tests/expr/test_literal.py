import ibis
from ibis.expr import datatypes
from ibis.expr.operations import Literal


def test_literal_equality_basic():
    a = ibis.literal(1).op()
    b = ibis.literal(1).op()

    assert a == b
    assert hash(a) == hash(b)


def test_literal_equality_int_float():
    # Note: This is different from the Python behavior for int/float comparison
    a = ibis.literal(1).op()
    b = ibis.literal(1.0).op()

    assert a != b


def test_literal_equality_int16_int32():
    # Note: This is different from the Python behavior for int/float comparison
    a = Literal(1, datatypes.int16)
    b = Literal(1, datatypes.int32)

    assert a != b


def test_literal_equality_int_interval():
    a = ibis.literal(1).op()
    b = ibis.interval(seconds=1).op()

    assert a != b


def test_literal_equality_interval():
    a = ibis.interval(seconds=1).op()
    b = ibis.interval(minutes=1).op()

    assert a != b

    # Currently these does't equal, but perhaps should be?
    c = ibis.interval(seconds=60).op()
    d = ibis.interval(minutes=1).op()

    assert c != d
