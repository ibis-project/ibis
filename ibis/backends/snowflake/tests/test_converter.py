from __future__ import annotations

import pyarrow as pa
import pytest

import ibis
from ibis.backends.snowflake.converter import (
    PYARROW_JSON_TYPE,
    SnowflakePyArrowData,
    source_schema,
)

JSON_ENCODED = ["json", "array<int64>", "map<string, int64>", "struct<a: int64>"]


@pytest.mark.parametrize("dtype", JSON_ENCODED)
def test_source_schema_sends_json_encoded_types_as_strings(dtype):
    # snowflake serializes VARIANT/ARRAY/OBJECT, so the nested arrow types the
    # ibis schema maps to never appear on the wire
    schema = ibis.schema({"x": dtype})
    assert source_schema(schema).field("x").type == pa.string()


def test_source_schema_accepts_what_the_connector_sends():
    # the batches path casts to this schema, so the cast must be a no-op for
    # the JSON-encoded columns and a widening for the narrow integer types
    # snowflake returns
    schema = ibis.schema({"i": "int64", "js": "json", "arr": "array<int64>"})
    raw = pa.table(
        {
            "i": pa.array([1, 2], pa.int8()),
            "js": pa.array(['{"a": 1}', "null"]),
            "arr": pa.array(["[1]", "[2]"]),
        }
    )
    cast = raw.cast(source_schema(schema))
    assert cast.schema.equals(source_schema(schema))
    assert cast["arr"].to_pylist() == ["[1]", "[2]"]


def test_empty_table_survives_json_wrapping():
    # the connector returns None for a zero-row result; the stand-in has to use
    # the wire types, because the extension wrapping rejects native nested
    # arrays as storage
    schema = ibis.schema(
        {"i": "int64", "arr": "array<int64>", "m": "map<string, int64>"}
    )
    empty = source_schema(schema).empty_table()
    converted = SnowflakePyArrowData.convert_table(empty, schema)
    assert len(converted) == 0
    assert converted.schema.field("arr").type == PYARROW_JSON_TYPE


def test_natively_typed_empty_table_would_not_survive():
    # the natively-typed stand-in raises, which is why the fix uses wire types
    schema = ibis.schema({"arr": "array<int64>"})
    with pytest.raises(TypeError, match="Incompatible storage type"):
        SnowflakePyArrowData.convert_table(schema.to_pyarrow().empty_table(), schema)


@pytest.mark.parametrize("dtype", JSON_ENCODED)
def test_eager_path_still_wraps_in_the_extension_type(dtype):
    # the display path is unchanged: `repr` depends on `as_py` parsing the JSON
    schema = ibis.schema({"x": dtype})
    raw = pa.table({"x": pa.array(['{"a": 1}'])})
    converted = SnowflakePyArrowData.convert_table(raw, schema)
    assert converted.schema.field("x").type == PYARROW_JSON_TYPE
    assert converted["x"][0].as_py() == {"a": 1}
