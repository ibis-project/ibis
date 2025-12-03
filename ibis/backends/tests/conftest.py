from __future__ import annotations

import sqlite3

import pytest
from packaging.version import parse as vparse

import ibis.common.exceptions as com
from ibis.backends.tests.errors import (
    ClickHouseDatabaseError,
    ImpalaHiveServer2Error,
    MySQLOperationalError,
    MySQLProgrammingError,
    PsycoPg2InternalError,
    Py4JJavaError,
    PySparkUnsupportedOperationException,
    TrinoUserError,
)


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

try:
    import pyspark

    pyspark_merge_exception = (
        PySparkUnsupportedOperationException
        if vparse(pyspark.__version__) >= vparse("3.5")
        else Py4JJavaError
    )
except ImportError:
    pyspark_merge_exception = None

NO_MERGE_SUPPORT_MARKS = [
    pytest.mark.notyet(
        ["clickhouse"],
        raises=ClickHouseDatabaseError,
        reason="MERGE INTO is not supported",
    ),
    pytest.mark.notyet(["datafusion"], reason="MERGE INTO is not supported"),
    pytest.mark.notyet(
        ["impala"],
        raises=ImpalaHiveServer2Error,
        reason="target table must be an Iceberg table",
    ),
    pytest.mark.notyet(
        ["mysql"], raises=MySQLProgrammingError, reason="MERGE INTO is not supported"
    ),
    pytest.mark.notimpl(["polars"], reason="`upsert` method not implemented"),
    pytest.mark.notyet(
        ["pyspark"],
        raises=pyspark_merge_exception,
        reason="MERGE INTO TABLE is not supported temporarily",
    ),
    pytest.mark.notyet(
        ["risingwave"],
        raises=PsycoPg2InternalError,
        reason="MERGE INTO is not supported",
    ),
    pytest.mark.notyet(
        ["sqlite"],
        raises=sqlite3.OperationalError,
        reason="MERGE INTO is not supported",
    ),
    pytest.mark.notyet(
        ["trino"],
        raises=TrinoUserError,
        reason="connector does not support modifying table rows",
    ),
]
NO_MERGE_SUPPORT = combine_marks(NO_MERGE_SUPPORT_MARKS)
