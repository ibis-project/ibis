import contextlib
import uuid

import pytest
import sqlalchemy.exc
from packaging.version import parse as vparse

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt

RAW_TEST_UUID = "08f48812-7948-4718-96c7-27fa6a398db6"
TEST_UUID = uuid.UUID(RAW_TEST_UUID)

SQLALCHEMY2 = vparse(sqlalchemy.__version__) >= vparse("2")

UUID_BACKEND_TYPE = {
    'bigquery': 'STRING',
    'duckdb': "UUID",
    'sqlite': "text",
    'snowflake': 'VARCHAR',
    'trino': 'varchar(32)' if SQLALCHEMY2 else 'uuid',
    "postgres": "uuid",
}

# TODO(krzysztof-kwitt): Should we unify it?
UUID_EXPECTED_VALUES = {
    'pandas': TEST_UUID,
    'bigquery': RAW_TEST_UUID,
    'duckdb': TEST_UUID,
    'sqlite': RAW_TEST_UUID,
    'snowflake': RAW_TEST_UUID,
    'trino': TEST_UUID if SQLALCHEMY2 else RAW_TEST_UUID,
    "postgres": TEST_UUID,
    'mysql': TEST_UUID if SQLALCHEMY2 else RAW_TEST_UUID,
    'mssql': TEST_UUID if SQLALCHEMY2 else RAW_TEST_UUID,
    'dask': TEST_UUID,
}


@pytest.mark.broken(
    ["duckdb"],
    '(duckdb.NotImplementedException) Not implemented Error: Unsupported type: "UUID"',
    raises=sqlalchemy.exc.NotSupportedError,
)
@pytest.mark.broken(
    ["pyspark"],
    "'UUID' object has no attribute '_get_object_id'",
    raises=AttributeError,
)
@pytest.mark.notimpl(
    ['impala', 'datafusion', 'polars', 'clickhouse'], raises=NotImplementedError
)
def test_uuid_literal(con, backend):
    backend_name = backend.name()

    expr = ibis.literal(RAW_TEST_UUID, type=dt.uuid)
    result = con.execute(expr)

    assert result == UUID_EXPECTED_VALUES[backend_name]

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == UUID_BACKEND_TYPE[backend_name]
