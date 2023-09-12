from __future__ import annotations

import hypothesis as h
import hypothesis.strategies as st
import pyarrow as pa
import pyarrow.tests.strategies as past
import pytest

import ibis.expr.datatypes as dt
from ibis.common.exceptions import IntegrityError
from ibis.formats.pyarrow import PyArrowSchema, PyArrowType


def assert_dtype_roundtrip(arrow_type, ibis_type=None, restored_type=None):
    dtype = PyArrowType.to_ibis(arrow_type, nullable=False)
    if ibis_type is not None:
        assert dtype == ibis_type

    patyp = PyArrowType.from_ibis(dtype)
    if restored_type is None:
        restored_type = arrow_type
    assert patyp == restored_type


roundtripable_types = st.deferred(
    lambda: (
        past.null_type
        | past.bool_type
        | past.integer_types
        | past.floating_types
        | past.duration_types
        | past.string_type
        | past.binary_type
        | past.timestamp_types
        | st.builds(pa.list_, roundtripable_types)
        | past.struct_types(roundtripable_types)
        | past.map_types(roundtripable_types, roundtripable_types)
    )
)


@h.given(roundtripable_types)
def test_roundtripable_types(arrow_type):
    assert_dtype_roundtrip(arrow_type)


@pytest.mark.parametrize(
    ("arrow_type", "ibis_type", "restored_type"),
    [
        (pa.decimal128(1, 1), dt.Decimal(1, 1, nullable=False), pa.decimal128(1, 1)),
        (pa.decimal128(10, 3), dt.Decimal(10, 3, nullable=False), pa.decimal128(10, 3)),
        (pa.decimal128(38, 3), dt.Decimal(38, 3, nullable=False), pa.decimal128(38, 3)),
        (pa.decimal256(1, 1), dt.Decimal(1, 1, nullable=False), pa.decimal128(1, 1)),
        (pa.decimal256(38, 5), dt.Decimal(38, 5, nullable=False), pa.decimal128(38, 5)),
        (pa.decimal256(39, 6), dt.Decimal(39, 6, nullable=False), pa.decimal256(39, 6)),
        (pa.decimal256(76, 6), dt.Decimal(76, 6, nullable=False), pa.decimal256(76, 6)),
        (pa.date32(), dt.Date(nullable=False), pa.date64()),
        (pa.date64(), dt.Date(nullable=False), pa.date64()),
        (pa.time32("s"), dt.Time(nullable=False), pa.time64("ns")),
        (pa.time32("ms"), dt.Time(nullable=False), pa.time64("ns")),
        (pa.time64("us"), dt.Time(nullable=False), pa.time64("ns")),
        (pa.time64("ns"), dt.Time(nullable=False), pa.time64("ns")),
        (pa.large_binary(), dt.Binary(nullable=False), pa.binary()),
        (pa.large_string(), dt.String(nullable=False), pa.string()),
        (
            pa.large_list(pa.int64()),
            dt.Array(dt.Int64(nullable=True), nullable=False),
            pa.list_(pa.int64()),
        ),
        (
            pa.list_(pa.int64(), list_size=3),
            dt.Array(dt.Int64(nullable=True), nullable=False),
            pa.list_(pa.int64()),
        ),
    ],
)
def test_non_roundtripable_types(arrow_type, ibis_type, restored_type):
    assert_dtype_roundtrip(arrow_type, ibis_type, restored_type)


@pytest.mark.parametrize("timezone", [None, "UTC"])
@pytest.mark.parametrize("nullable", [True, False])
def test_timestamp_no_scale(timezone, nullable):
    dtype = dt.Timestamp(scale=None, timezone=timezone, nullable=nullable)
    assert dtype.to_pyarrow() == pa.timestamp("us", tz=timezone)


def test_month_day_nano_type_unsupported():
    with pytest.raises(ValueError, match="Arrow interval type is not supported"):
        PyArrowType.to_ibis(pa.month_day_nano_interval())


@pytest.mark.parametrize("value_nullable", [True, False])
def test_dtype_from_nullable_map_type(value_nullable):
    # the key field cannot be nullable
    pyarrow_type = pa.map_(
        pa.int64(), pa.field("value", pa.int64(), nullable=value_nullable)
    )
    ibis_type = PyArrowType.to_ibis(pyarrow_type)
    restored_type = PyArrowType.from_ibis(ibis_type)

    assert ibis_type == dt.Map(
        dt.Int64(nullable=False), dt.Int64(nullable=value_nullable)
    )
    assert restored_type.key_field.type == pa.int64()
    assert restored_type.key_field.nullable is False
    assert restored_type.item_field.type == pa.int64()
    assert restored_type.item_field.nullable is value_nullable


@pytest.mark.parametrize("value_nullable", [True, False])
@pytest.mark.parametrize("list_nullable", [True, False])
def test_dtype_from_nullable_list_type(value_nullable, list_nullable):
    pyarrow_type = pa.list_(pa.field("value", pa.int64(), nullable=value_nullable))
    ibis_type = PyArrowType.to_ibis(pyarrow_type, nullable=list_nullable)
    restored_type = PyArrowType.from_ibis(ibis_type)

    assert ibis_type == dt.Array(
        dt.Int64(nullable=value_nullable), nullable=list_nullable
    )
    assert restored_type.value_field.type == pa.int64()
    assert restored_type.value_field.nullable is value_nullable


@pytest.mark.parametrize(
    ("ibis_type", "arrow_type"),
    [
        (dt.Set(dt.String(nullable=True)), pa.list_(pa.string())),
        (
            dt.Set(dt.String(nullable=False)),
            pa.list_(pa.field("item", pa.string(), nullable=False)),
        ),
    ],
)
def test_ibis_exclusive_types(ibis_type, arrow_type):
    assert PyArrowType.from_ibis(ibis_type) == arrow_type


def test_schema_from_pyarrow_checks_duplicate_column_names():
    arrow_schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("a", pa.int64()),
        ]
    )
    with pytest.raises(IntegrityError, match="Duplicate column name"):
        PyArrowSchema.to_ibis(arrow_schema)


@h.given(past.schemas(roundtripable_types))
def test_schema_roundtrip(pyarrow_schema):
    unique_column_names = set(pyarrow_schema.names)
    h.assume(len(unique_column_names) == len(pyarrow_schema.names))

    ibis_schema = PyArrowSchema.to_ibis(pyarrow_schema)
    restored = PyArrowSchema.from_ibis(ibis_schema)
    assert pyarrow_schema.equals(restored)


def test_unknown_dtype_gets_converted_to_string():
    assert PyArrowType.from_ibis(dt.unknown) == pa.string()
