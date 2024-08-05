from __future__ import annotations

import contextlib
import uuid

import pyarrow as pa
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt

TEST_UUID = "08f48812-7948-4718-96c7-27fa6a398db6"

UUID_BACKEND_TYPE = {
    "bigquery": "STRING",
    "clickhouse": "Nullable(UUID)",
    "duckdb": "UUID",
    "exasol": "UUID",
    "flink": "CHAR(36) NOT NULL",
    "impala": "STRING",
    "mssql": "uniqueidentifier",
    "postgres": "uuid",
    "risingwave": "character varying",
    "snowflake": "VARCHAR",
    "sqlite": "text",
    "trino": "uuid",
}


@pytest.mark.notimpl(["polars"], raises=NotImplementedError)
def test_uuid_literal(con, backend):
    backend_name = backend.name()

    expr = ibis.literal(TEST_UUID, type=dt.uuid)
    result = con.execute(expr)

    assert result == TEST_UUID

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == UUID_BACKEND_TYPE[backend_name]


@pytest.mark.notimpl(["polars"], raises=NotImplementedError)
def test_uuid_to_pyarrow(con):
    expr = ibis.literal(TEST_UUID, type=dt.uuid)
    result = con.to_pyarrow(expr)
    assert str(result) == TEST_UUID

    tbl = con.tables.functional_alltypes.mutate(uuid=expr).select("uuid")
    result = con.to_pyarrow(tbl.limit(2))
    assert pa.types.is_string(result["uuid"].type)


@pytest.mark.notimpl(
    ["druid", "exasol", "oracle", "polars", "pyspark", "risingwave", "pandas", "dask"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.never(
    ["mysql"], raises=AssertionError, reason="MySQL generates version 1 UUIDs"
)
def test_uuid_function(con):
    obj = con.execute(ibis.uuid())
    assert isinstance(obj, str)
    assert uuid.UUID(obj).version == 4


@pytest.mark.notimpl(
    ["druid", "exasol", "oracle", "polars", "pyspark", "risingwave", "pandas", "dask"],
    raises=com.OperationNotDefinedError,
)
def test_uuid_unique_each_row(con):
    expr = (
        con.tables.functional_alltypes.mutate(uuid=ibis.uuid()).limit(2).uuid.nunique()
    )
    assert expr.execute() == 2
