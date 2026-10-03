from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.snowflake.tests.conftest import _get_url
from ibis.backends.sql.datatypes import SnowflakeType
from ibis.util import gen_name

dtypes = [
    ("FIXED", dt.int64),
    ("REAL", dt.float64),
    ("TEXT", dt.string),
    ("DATE", dt.date),
    ("TIMESTAMP", dt.Timestamp(scale=9)),
    ("VARIANT", dt.json),
    ("TIMESTAMP_LTZ", dt.Timestamp(timezone="UTC", scale=9)),
    ("TIMESTAMP_TZ", dt.Timestamp(timezone="UTC", scale=9)),
    ("TIMESTAMP_NTZ", dt.Timestamp(scale=9)),
    ("OBJECT", dt.Map(dt.string, dt.json)),
    ("ARRAY", dt.Array(dt.json)),
    ("BINARY", dt.binary),
    ("TIME", dt.time),
    ("BOOLEAN", dt.boolean),
    # VECTOR(<element>, <dimension>): fixed-length numeric vectors.
    # https://docs.snowflake.com/en/sql-reference/data-types-vector
    # FLOAT/INT inside VECTOR are documented as 32-bit, so the element
    # type narrows to Float32/Int32 even though scalar Snowflake FLOAT
    # otherwise resolves to Float64.
    ("VECTOR(FLOAT, 4)", dt.Array(dt.Float32(nullable=False), length=4)),
    ("VECTOR(FLOAT, 512)", dt.Array(dt.Float32(nullable=False), length=512)),
    ("VECTOR(INT, 8)", dt.Array(dt.Int32(nullable=False), length=8)),
]


@pytest.mark.parametrize(
    ("snowflake_type", "ibis_type"),
    [
        param(snowflake_type, ibis_type, id=snowflake_type)
        for snowflake_type, ibis_type in dtypes
    ],
)
def test_parse(snowflake_type, ibis_type):
    assert SnowflakeType.from_string(snowflake_type.upper()) == ibis_type


@pytest.fixture(scope="module")
def con():
    return ibis.connect(_get_url())


user_dtypes = [
    ("NUMBER", dt.int64),
    ("DECIMAL", dt.int64),
    ("NUMERIC", dt.int64),
    ("NUMBER(5)", dt.int64),
    ("DECIMAL(5, 2)", dt.Decimal(5, 2)),
    ("NUMERIC(21, 17)", dt.Decimal(21, 17)),
    ("INT", dt.int64),
    ("INTEGER", dt.int64),
    ("BIGINT", dt.int64),
    ("SMALLINT", dt.int64),
    ("TINYINT", dt.int64),
    ("BYTEINT", dt.int64),
    ("FLOAT", dt.float64),
    ("FLOAT4", dt.float64),
    ("FLOAT8", dt.float64),
    ("DOUBLE", dt.float64),
    ("DOUBLE PRECISION", dt.float64),
    ("REAL", dt.float64),
    ("VARCHAR", dt.string),
    ("VARCHAR(50)", dt.String(length=50)),
    ("CHAR", dt.String(length=1)),
    ("CHAR(5)", dt.String(length=5)),
    ("CHARACTER", dt.String(length=1)),
    ("STRING", dt.string),
    ("TEXT", dt.string),
    ("BINARY", dt.binary),
    ("VARBINARY", dt.binary),
    ("VARBINARY(8388608)", dt.binary),
    ("BOOLEAN", dt.boolean),
    ("DATE", dt.date),
    ("TIME", dt.time),
    ("VARIANT", dt.json),
    ("OBJECT", dt.Map(dt.string, dt.json)),
    ("ARRAY", dt.Array(dt.json)),
    # VECTOR round-trip via CREATE TEMP TABLE: exercises the full path
    # from Snowflake's information-schema-style metadata back through
    # ``SnowflakeType.from_string``.
    ("VECTOR(FLOAT, 4)", dt.Array(dt.Float32(nullable=False), length=4)),
    ("VECTOR(INT, 8)", dt.Array(dt.Int32(nullable=False), length=8)),
]


@pytest.mark.parametrize(
    ("snowflake_type", "ibis_type"),
    [
        param(snowflake_type, ibis_type, id=snowflake_type)
        for snowflake_type, ibis_type in user_dtypes
    ],
)
def test_extract_type_from_table_query(con, snowflake_type, ibis_type):
    name = gen_name("test_extract_type_from_table")
    query = f'CREATE TEMP TABLE "{name}" ("a" {snowflake_type})'
    con.raw_sql(query).close()
    expected_schema = ibis.schema(dict(a=ibis_type))

    t = con.sql(f'SELECT "a" FROM "{name}"')
    assert t.schema() == expected_schema


@pytest.mark.parametrize(
    ("snowflake_type", "ibis_type"),
    [
        param("DATETIME", dt.Timestamp(scale=9)),
        param("TIMESTAMP", dt.Timestamp(scale=9)),
        param("TIMESTAMP(3)", dt.Timestamp(scale=3)),
        param("TIMESTAMP_LTZ", dt.Timestamp(timezone="UTC", scale=9)),
        param("TIMESTAMP_LTZ(3)", dt.Timestamp(timezone="UTC", scale=3)),
        param("TIMESTAMP_NTZ", dt.Timestamp(scale=9)),
        param("TIMESTAMP_NTZ(3)", dt.Timestamp(scale=3)),
        param("TIMESTAMP_TZ", dt.Timestamp(timezone="UTC", scale=9)),
        param("TIMESTAMP_TZ(3)", dt.Timestamp(timezone="UTC", scale=3)),
    ],
)
def test_extract_timestamp_from_table(con, snowflake_type, ibis_type):
    name = gen_name("test_extract_type_from_table")
    query = f'CREATE TEMP TABLE "{name}" ("a" {snowflake_type})'
    con.raw_sql(query).close()

    expected_schema = ibis.schema(dict(a=ibis_type))

    t = con.table(name)
    assert t.schema() == expected_schema


compiled_vector_dtypes = [
    param(dt.Array(dt.Float32(nullable=False), length=4), "VECTOR(FLOAT, 4)", id="f32"),
    param(dt.Array(dt.Int32(nullable=False), length=8), "VECTOR(INT, 8)", id="i32"),
    # Snowflake VECTOR elements are 32-bit only, so wider element types narrow.
    param(dt.Array(dt.float64, length=3), "VECTOR(FLOAT, 3)", id="f64-narrows"),
    param(dt.Array(dt.int64, length=3), "VECTOR(INT, 3)", id="i64-narrows"),
]


@pytest.mark.parametrize(("ibis_type", "snowflake_type"), compiled_vector_dtypes)
def test_compile_fixed_length_numeric_array_to_vector(ibis_type, snowflake_type):
    """Fixed-length numeric arrays compile back to VECTOR.

    The parse direction is covered by ``test_parse``; this is the other half, and
    unlike ``test_extract_type_from_table_query`` it needs no Snowflake connection.

    Note the last two cases: VECTOR only has 32-bit element types, so a 64-bit element
    narrows rather than erroring. That is the only way to express such an array in
    Snowflake at all, but it is silent, so it is pinned here deliberately.
    """
    assert SnowflakeType.to_string(ibis_type) == snowflake_type


non_vector_dtypes = [
    param(dt.Array(dt.string), id="variable-length-string"),
    param(dt.Array(dt.float32), id="variable-length-float"),
    param(dt.Array(dt.string, length=4), id="fixed-length-string"),
    param(dt.Array(dt.Struct({"a": dt.int64}), length=2), id="fixed-length-struct"),
    param(dt.Array(dt.Array(dt.int64), length=2), id="fixed-length-nested-array"),
]


@pytest.mark.parametrize("ibis_type", non_vector_dtypes)
def test_non_numeric_and_variable_length_arrays_still_compile_to_array(ibis_type):
    """Only fixed-length *numeric* arrays may become VECTOR.

    Snowflake's VECTOR accepts INT and FLOAT elements only, so emitting VECTOR for a
    fixed-length array of any other element type would produce invalid DDL. Variable-
    length arrays must keep compiling to the existing JSON-backed ARRAY.
    """
    assert SnowflakeType.to_string(ibis_type) == "ARRAY"


def test_vector_with_non_numeric_element_falls_back_to_the_default_mapping():
    """A VECTOR whose element is neither INT nor FLOAT uses the ordinary scalar mapping.

    Snowflake will not produce one -- VECTOR is documented as INT or FLOAT only -- so
    this is the defensive branch of the parser rather than a shape seen in the wild.

    It is deliberately not round-trippable: the result compiles back to ARRAY, not
    VECTOR, because Snowflake has no way to express a fixed-length string vector.
    """
    parsed = SnowflakeType.from_string("VECTOR(VARCHAR, 4)")

    assert parsed == dt.Array(dt.string, length=4)
    assert SnowflakeType.to_string(parsed) == "ARRAY"


def test_vector_scalar_pyarrow_passthrough_for_fixed_size_arrays():
    """``convert_scalar`` needs the same guard as ``convert_column``.

    A single VECTOR value arrives as a native fixed-length pyarrow scalar, so routing it
    through the JSON-extension wrap fails the same way the column path did.
    """
    import pyarrow as pa

    from ibis.backends.snowflake.converter import SnowflakePyArrowData

    scalar = pa.scalar([1.0, 2.0, 3.0, 4.0], type=pa.list_(pa.float32(), list_size=4))
    dtype = dt.Array(dt.Float32(nullable=False), length=4)

    result = SnowflakePyArrowData.convert_scalar(scalar, dtype)

    assert result is scalar, "fixed-size-list scalars should pass through unchanged"
    assert result.as_py() == [1.0, 2.0, 3.0, 4.0]


def test_vector_column_pyarrow_passthrough_for_fixed_size_arrays():
    """``SnowflakePyArrowData.convert_column`` must pass fixed-length array
    columns through unchanged.

    Snowflake's VECTOR(<element>, <dimension>) deserializes natively to
    pyarrow ``fixed_size_list<element>[length]`` via the Snowflake Python
    connector. Without the fixed-size-list pass-through, the converter
    routes the column through the JSON-extension wrapping path -- which
    raises ``ArrowNotImplementedError: Unsupported cast from
    fixed_size_list<...> to utf8`` because the storage type of
    ``PYARROW_JSON_TYPE`` is ``utf8``.

    This test exercises the converter directly with a synthetic
    fixed-size-list pyarrow column and the matching
    ``Array(Float32, length=N)`` ibis dtype, so it runs without any
    Snowflake connection.
    """
    import pyarrow as pa

    from ibis.backends.snowflake.converter import SnowflakePyArrowData

    column = pa.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        type=pa.list_(pa.float32(), list_size=4),
    )
    dtype = dt.Array(dt.Float32(nullable=False), length=4)

    result = SnowflakePyArrowData.convert_column(column, dtype)

    assert result is column, "fixed-size-list columns should pass through unchanged"
    assert result.to_pylist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_array_discovery(con):
    t = con.tables.ARRAY_TYPES
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.json),
            y=dt.Array(dt.json),
            z=dt.Array(dt.json),
            grouper=dt.string,
            scalar_column=dt.float64,
            multi_dim=dt.Array(dt.json),
        )
    )
    assert t.schema() == expected
