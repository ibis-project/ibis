from __future__ import annotations

import contextlib

import pytest

import ibis
import ibis.common.exceptions as com

BINARY_BACKEND_TYPES = {
    "bigquery": "BYTES",
    "clickhouse": "String",
    "duckdb": "BLOB",
    "snowflake": "BINARY",
    "sqlite": "blob",
    "trino": "varbinary",
    "postgres": "bytea",
    "risingwave": "bytea",
    "flink": "BINARY(1) NOT NULL",
}


@pytest.mark.notimpl(
    ["clickhouse", "impala", "druid", "oracle"],
    "Unsupported type: Binary(nullable=True)",
    raises=NotImplementedError,
)
@pytest.mark.notimpl(
    ["exasol"],
    "Exasol does not have native support for a binary data type.",
    raises=NotImplementedError,
)
def test_binary_literal(con, backend):
    expr = ibis.literal(b"A")
    result = con.execute(expr)
    assert result == b"A"

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == BINARY_BACKEND_TYPES[backend_name]
