from __future__ import annotations

import contextlib
import uuid

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends.tests.errors import PyAthenaOperationalError

TEST_UUID_STR = "08f48812-7948-4718-96c7-27fa6a398db6"
TEST_UUID = uuid.UUID(TEST_UUID_STR)

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
    "databricks": "string",
}


@pytest.mark.notimpl(["polars"], raises=NotImplementedError)
@pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            lambda: ibis.literal(TEST_UUID_STR, type=dt.uuid), id="literal_str"
        ),
        pytest.param(
            lambda: ibis.literal(TEST_UUID, type=dt.uuid), id="literal_uuid_typed"
        ),
        pytest.param(lambda: ibis.literal(TEST_UUID), id="literal_uuid_untyped"),
        pytest.param(lambda: ibis.uuid(TEST_UUID_STR), id="uuid_str"),
        pytest.param(lambda: ibis.uuid(TEST_UUID), id="uuid_uuid"),
    ],
)
def test_uuid_literal(con, backend, value):
    backend_name = backend.name()

    expr = value()
    result = con.execute(expr)

    assert result == TEST_UUID

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == UUID_BACKEND_TYPE[backend_name]


@pytest.mark.notimpl(
    ["druid", "exasol", "oracle", "polars", "risingwave", "pyspark"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError)
@pytest.mark.never(
    ["mysql"], raises=AssertionError, reason="MySQL generates version 1 UUIDs"
)
def test_uuid_function(con):
    obj = con.execute(ibis.uuid())
    assert isinstance(obj, uuid.UUID)
    assert obj.version == 4


@pytest.mark.notimpl(
    ["druid", "exasol", "oracle", "polars", "risingwave", "pyspark"],
    raises=com.OperationNotDefinedError,
)
def test_uuid_unique_each_row(con):
    expr = (
        con.tables.functional_alltypes.mutate(uuid=ibis.uuid()).limit(2).uuid.nunique()
    )
    assert expr.execute() == 2
