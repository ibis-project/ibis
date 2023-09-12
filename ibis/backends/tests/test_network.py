from __future__ import annotations

import contextlib
from ipaddress import IPv4Address, IPv6Address

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt

MACADDR_BACKEND_TYPE = {
    "bigquery": "STRING",
    "clickhouse": "String",
    "duckdb": "VARCHAR",
    "snowflake": "VARCHAR",
    "sqlite": "text",
    "trino": "varchar(17)",
    "impala": "STRING",
    "postgres": "text",
}


@pytest.mark.notimpl(["polars"], raises=NotImplementedError)
def test_macaddr_literal(con, backend):
    test_macaddr = "00:00:0A:BB:28:FC"
    expr = ibis.literal(test_macaddr, type=dt.macaddr)
    result = con.execute(expr)
    assert result == test_macaddr

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == MACADDR_BACKEND_TYPE[backend_name]


@pytest.mark.parametrize(
    ("test_value", "expected_values", "expected_types"),
    [
        param(
            "127.0.0.1",
            {
                "bigquery": "127.0.0.1",
                "clickhouse": IPv4Address("127.0.0.1"),
                "duckdb": "127.0.0.1",
                "snowflake": "127.0.0.1",
                "sqlite": "127.0.0.1",
                "trino": "127.0.0.1",
                "impala": "127.0.0.1",
                "postgres": "127.0.0.1",
                "pandas": "127.0.0.1",
                "pyspark": "127.0.0.1",
                "mysql": "127.0.0.1",
                "dask": "127.0.0.1",
                "mssql": "127.0.0.1",
                "datafusion": "127.0.0.1",
            },
            {
                "bigquery": "STRING",
                "clickhouse": "IPv4",
                "duckdb": "VARCHAR",
                "snowflake": "VARCHAR",
                "sqlite": "text",
                "trino": "varchar(9)",
                "impala": "STRING",
                "postgres": "text",
            },
            id="ipv4",
        ),
        param(
            "2001:db8::1",
            {
                "bigquery": "2001:db8::1",
                "clickhouse": IPv6Address("2001:db8::1"),
                "duckdb": "2001:db8::1",
                "snowflake": "2001:db8::1",
                "sqlite": "2001:db8::1",
                "trino": "2001:db8::1",
                "impala": "2001:db8::1",
                "postgres": "2001:db8::1",
                "pandas": "2001:db8::1",
                "pyspark": "2001:db8::1",
                "mysql": "2001:db8::1",
                "dask": "2001:db8::1",
                "mssql": "2001:db8::1",
                "datafusion": "2001:db8::1",
            },
            {
                "bigquery": "STRING",
                "clickhouse": "IPv6",
                "duckdb": "VARCHAR",
                "snowflake": "VARCHAR",
                "sqlite": "text",
                "trino": "varchar(11)",
                "impala": "STRING",
                "postgres": "text",
            },
            id="ipv6",
        ),
    ],
)
@pytest.mark.notimpl(["polars"], raises=NotImplementedError)
@pytest.mark.notimpl(["druid", "oracle"], raises=KeyError)
def test_inet_literal(con, backend, test_value, expected_values, expected_types):
    backend_name = backend.name()
    expr = ibis.literal(test_value, type=dt.inet)
    result = con.execute(expr)

    assert result == expected_values[backend_name]

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == expected_types[backend_name]
