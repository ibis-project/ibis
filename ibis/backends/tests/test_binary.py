from __future__ import annotations

import contextlib

import pytest
import sqlalchemy.exc

import ibis
import ibis.common.exceptions as com

BINARY_BACKEND_TYPES = {
    "bigquery": "BYTES",
    "clickhouse": "String",
    "duckdb": "BLOB",
    "snowflake": "BINARY",
    "sqlite": "blob",
    "trino": "STRING",
    "postgres": "bytea",
}


@pytest.mark.broken(
    ["trino"],
    "(builtins.AttributeError) 'bytes' object has no attribute 'encode'",
    raises=sqlalchemy.exc.StatementError,
)
@pytest.mark.broken(
    ["clickhouse", "impala"],
    "Unsupported type: Binary(nullable=True)",
    raises=NotImplementedError,
)
def test_binary_literal(con, backend):
    expr = ibis.literal(b"A")
    result = con.execute(expr)
    assert result == b"A"

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == BINARY_BACKEND_TYPES[backend_name]
