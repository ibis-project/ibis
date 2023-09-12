from __future__ import annotations

import datetime
import decimal
import uuid

import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.common.collections import frozendict
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
    a = Literal(1, dt.int16)
    b = Literal(1, dt.int32)

    assert a != b


def test_literal_equality_int_interval():
    a = ibis.literal(1).op()
    b = ibis.interval(seconds=1).op()

    assert a != b


def test_literal_equality_interval():
    a = ibis.interval(seconds=1).op()
    b = ibis.interval(minutes=1).op()

    assert a != b

    # Currently these don't equal, but perhaps should?
    c = ibis.interval(seconds=60).op()
    d = ibis.interval(minutes=1).op()

    assert c != d


def test_pickle_literal():
    a = Literal(1, dt.int16)
    b = Literal(1, dt.int32)

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
        pytest.param(str(uuid.uuid1()), "uuid", uuid.UUID, id="str_uuid1_as_uuid"),
        pytest.param(str(uuid.uuid4()), "uuid", uuid.UUID, id="str_uuid4_as_uuid"),
        pytest.param(uuid.uuid1(), "string", str, id="uuid1_as_str"),
        pytest.param(uuid.uuid4(), "string", str, id="uuid4_as_str"),
        pytest.param(str(uuid.uuid1()), "string", str, id="str_uuid1_as_str"),
        pytest.param(str(uuid.uuid4()), "string", str, id="str_uuid4_as_str"),
        pytest.param(0, "float", float, id="int_zero_as_float"),
        pytest.param(0.0, "float", float, id="float_zero_as_float"),
        pytest.param(0.0, "float32", float, id="float32_zero_as_float"),
        pytest.param(42, "float", float, id="int_as_float"),
        pytest.param(42.0, "float", float, id="float_as_float"),
        pytest.param(42.0, None, float, id="float_implicit_type_as_float"),
    ],
)
def test_normalized_underlying_value(userinput, literal_type, expected_type):
    a = ibis.literal(userinput, type=literal_type)

    assert isinstance(a.op().value, expected_type)


@pytest.mark.parametrize(
    "value",
    [
        dict(field1="value1", field2=3.14),
        dict(field1="value1", field2="3.14"),  # coerceable type
        dict(field1="value1", field2=1),  # coerceable type
        dict(field2=2.72, field1="value1"),  # wrong field order
        dict(field1="value1", field2=3.14, field3="extra"),  # extra field
    ],
)
def test_struct_literal(value):
    typestr = "struct<field1: string, field2: float64>"
    a = ibis.struct(value, type=typestr)
    assert a.op().value == frozendict(
        field1=str(value["field1"]), field2=float(value["field2"])
    )
    assert a.type() == dt.dtype(typestr)


@pytest.mark.parametrize(
    "value",
    [
        dict(field1="value1", field3=3.14),  # wrong field name
        dict(field1="value1"),  # missing field
    ],
)
def test_struct_literal_non_castable(value):
    typestr = "struct<field1: string, field2: float64>"
    with pytest.raises(TypeError, match="Unable to normalize"):
        ibis.struct(value, type=typestr)


def test_struct_cast_to_empty_struct():
    value = ibis.struct({"a": 1, "b": 2.0})
    assert value.type().castable(dt.Struct({}))


@pytest.mark.parametrize(
    "value",
    [
        dict(key1="value1", key2="value2"),
    ],
)
def test_map_literal(value):
    typestr = "map<string, int8>"
    a = ibis.map(["a", "b"], [1, 2])
    assert a.op().keys.value == ("a", "b")
    assert a.op().values.value == (1, 2)
    assert a.type() == dt.dtype(typestr)


@pytest.mark.parametrize(
    "value",
    [
        dict(key1="value1", key2=6.25),  # heterogeneous map values
    ],
)
def test_map_literal_non_castable(value):
    typestr = "map<string, string>"
    with pytest.raises(TypeError):
        ibis.map(value, type=typestr)


def test_literal_mixed_type_fails():
    data = [1, "a"]
    with pytest.raises(TypeError, match="Cannot compute precedence"):
        ibis.literal(data)


def test_timestamp_literal_without_tz():
    now_raw = datetime.datetime.utcnow()
    assert now_raw.tzinfo is None
    assert ibis.literal(now_raw).type().timezone is None


def test_integer_as_decimal():
    lit = ibis.literal(12, type="decimal")
    assert lit.op().value == decimal.Decimal(12)
