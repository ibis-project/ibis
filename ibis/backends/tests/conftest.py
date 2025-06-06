from __future__ import annotations

import pytest

import ibis.common.exceptions as com
from ibis.backends.tests.errors import MySQLOperationalError


def combine_marks(marks: list) -> callable:
    def decorator(func):
        for mark in reversed(marks):
            func = mark(func)
        return func

    return decorator


NO_ARRAY_SUPPORT_MARKS = [
    pytest.mark.never(
        ["sqlite", "mysql", "exasol"], reason="No array support", raises=Exception
    ),
    pytest.mark.never(
        ["mssql"],
        reason="No array support",
        raises=(
            com.UnsupportedBackendType,
            com.OperationNotDefinedError,
            AssertionError,
        ),
    ),
    pytest.mark.never(
        ["mysql"],
        reason="No array support",
        raises=(
            com.UnsupportedBackendType,
            com.OperationNotDefinedError,
            MySQLOperationalError,
        ),
    ),
    pytest.mark.notyet(
        ["impala"],
        reason="No array support",
        raises=(
            com.UnsupportedBackendType,
            com.OperationNotDefinedError,
            com.TableNotFound,
        ),
    ),
    pytest.mark.notimpl(["druid", "oracle"], raises=Exception),
]
NO_ARRAY_SUPPORT = combine_marks(NO_ARRAY_SUPPORT_MARKS)


NO_STRUCT_SUPPORT_MARKS = [
    pytest.mark.never(["mysql", "sqlite", "mssql"], reason="No struct support"),
    pytest.mark.notyet(["impala"]),
    pytest.mark.notimpl(["druid", "oracle", "exasol"]),
]
NO_STRUCT_SUPPORT = combine_marks(NO_STRUCT_SUPPORT_MARKS)

NO_MAP_SUPPORT_MARKS = [
    pytest.mark.never(
        ["sqlite", "mysql", "mssql"], reason="Unlikely to ever add map support"
    ),
    pytest.mark.notyet(
        ["bigquery", "impala"], reason="Backend doesn't yet implement map types"
    ),
    pytest.mark.notimpl(
        ["exasol", "polars", "druid", "oracle"],
        reason="Not yet implemented in ibis",
    ),
]
NO_MAP_SUPPORT = combine_marks(NO_MAP_SUPPORT_MARKS)

NO_JSON_SUPPORT_MARKS = [
    pytest.mark.never(["impala"], reason="doesn't support JSON and never will"),
    pytest.mark.notyet(["clickhouse"], reason="upstream is broken"),
    pytest.mark.notimpl(["datafusion", "exasol", "mssql", "druid", "oracle"]),
]
NO_JSON_SUPPORT = combine_marks(NO_JSON_SUPPORT_MARKS)
