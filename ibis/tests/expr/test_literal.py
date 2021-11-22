import uuid

import pytest

import ibis
from ibis.expr import datatypes
from ibis.expr.operations import Literal
from ibis.tests.util import assert_pickle_roundtrip


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


def test_pickle_literal():
    a = Literal(1, datatypes.int16)
    b = Literal(1, datatypes.int32)

    assert_pickle_roundtrip(a)
    assert_pickle_roundtrip(b)


def test_pickle_literal_interval():
    a = ibis.interval(seconds=1).op()

    assert_pickle_roundtrip(a)


@pytest.mark.parametrize(
    ("userinput", "literal_type", "expected_type"),
    [
        pytest.param(uuid.uuid1(), "uuid", uuid.UUID, id="uuid1_as_uuid"),
        pytest.param(uuid.uuid4(), "uuid", uuid.UUID, id="uuid4_as_uuid"),
        pytest.param(
            str(uuid.uuid1()), "uuid", uuid.UUID, id="str_uuid1_as_uuid"
        ),
        pytest.param(
            str(uuid.uuid4()), "uuid", uuid.UUID, id="str_uuid4_as_uuid"
        ),
        pytest.param(uuid.uuid1(), "string", str, id="uuid1_as_str"),
        pytest.param(uuid.uuid4(), "string", str, id="uuid4_as_str"),
        pytest.param(str(uuid.uuid1()), "string", str, id="str_uuid1_as_str"),
        pytest.param(str(uuid.uuid4()), "string", str, id="str_uuid4_as_str"),
        pytest.param(0, "float", float, id="int_zero_as_float"),
        pytest.param(0.0, "float", float, id="float_zero_as_float"),
        pytest.param(42, "float", float, id="int_as_float"),
        pytest.param(42.0, "float", float, id="float_as_float"),
        pytest.param(42.0, None, float, id="float_implicit_type_as_float"),
    ],
)
def test_normalized_underlying_value(userinput, literal_type, expected_type):
    a = ibis.literal(userinput, type=literal_type)

    assert isinstance(a.op().value, expected_type)
