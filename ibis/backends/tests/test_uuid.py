from __future__ import annotations

import contextlib
import uuid

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt

RAW_TEST_UUID = "08f48812-7948-4718-96c7-27fa6a398db6"
TEST_UUID = uuid.UUID(RAW_TEST_UUID)

UUID_BACKEND_TYPE = {
    "bigquery": "STRING",
    "clickhouse": "Nullable(UUID)",
    "duckdb": "UUID",
    "exasol": "UUID",
    "flink": "CHAR(36) NOT NULL",
    "impala": "STRING",
    "postgres": "uuid",
    "risingwave": "character varying",
    "snowflake": "VARCHAR",
    "sqlite": "text",
    "trino": "uuid",
}


@pytest.mark.notimpl(["polars"], raises=NotImplementedError)
def test_uuid_literal(con, backend):
    backend_name = backend.name()

    expr = ibis.literal(RAW_TEST_UUID, type=dt.uuid)
    result = con.execute(expr)

    assert result == TEST_UUID

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == UUID_BACKEND_TYPE[backend_name]


@pytest.mark.notimpl(
    [
        "datafusion",
        "druid",
        "exasol",
        "flink",
        "mssql",
        "mysql",
        "oracle",
        "polars",
        "pyspark",
        "risingwave",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["pandas", "dask"], raises=ValueError)
def test_uuid_function(con):
    obj = con.execute(ibis.uuid())
    assert isinstance(obj, uuid.UUID)
    assert obj.version == 4
