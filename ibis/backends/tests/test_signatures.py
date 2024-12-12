from __future__ import annotations

import inspect

import pytest

from ibis.backends import (
    BaseBackend,
    CanCreateCatalog,
    CanCreateDatabase,
    CanListCatalog,
    CanListDatabase,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.tests.signature.typecheck import compatible

SKIP_METHODS = ["do_connect", "from_connection"]


def _scrape_methods(modules, params):
    params = []
    for module in modules:
        methods = list(filter(lambda x: not x.startswith("_"), dir(module)))
        for method in methods:
            # Only test methods that are callable (so we can grab the signature)
            # and skip any methods that we don't want to check
            if (
                method not in SKIP_METHODS
                and method not in marks.keys()
                and callable(getattr(module, method))
            ):
                params.append((module, method))

    return params


marks = {
    "compile": pytest.param(
        BaseBackend,
        "compile",
        marks=pytest.mark.notyet(
            [
                "bigquery",
                "clickhouse",
                "datafusion",
                "druid",
                "duckdb",
                "exasol",
                "impala",
                "mssql",
                "mysql",
                "oracle",
                "postgres",
                "pyspark",
                "risingwave",
                "snowflake",
                "sqlite",
                "trino",
                "databricks",
            ],
            reason="SQL backends all have an additional `pretty` argument for formatting the generated SQL",
        ),
    ),
    "create_database": pytest.param(
        CanCreateDatabase,
        "create_database",
        marks=pytest.mark.notyet(["clickhouse", "flink", "impala", "mysql", "pyspark"]),
    ),
    "drop_database": pytest.param(
        CanCreateDatabase,
        "drop_database",
        marks=pytest.mark.notyet(["clickhouse", "impala", "mysql", "pyspark"]),
    ),
    "drop_table": pytest.param(
        SQLBackend,
        "drop_table",
        marks=pytest.mark.notyet(["bigquery", "druid", "flink", "impala", "polars"]),
    ),
    "execute": pytest.param(
        SQLBackend,
        "execute",
        marks=pytest.mark.notyet(["clickhouse", "datafusion", "flink", "mysql"]),
    ),
    "insert": pytest.param(
        SQLBackend,
        "insert",
        marks=pytest.mark.notyet(["clickhouse", "flink", "impala"]),
    ),
    "list_databases": pytest.param(
        CanCreateDatabase,
        "list_databases",
        marks=pytest.mark.notyet(
            [
                "clickhouse",
                "flink",
                "impala",
                "mysql",
                "postgres",
                "risingwave",
                "sqlite",
            ]
        ),
    ),
    "list_tables": pytest.param(
        BaseBackend,
        "list_tables",
        marks=pytest.mark.notyet(["flink"]),
    ),
    "read_csv": pytest.param(
        BaseBackend,
        "read_csv",
        marks=pytest.mark.notyet(["duckdb", "flink", "pyspark", "datafusion"]),
    ),
    "read_delta": pytest.param(
        BaseBackend,
        "read_delta",
        marks=pytest.mark.notyet(["datafusion", "duckdb", "polars", "pyspark"]),
    ),
    "read_json": pytest.param(
        BaseBackend,
        "read_json",
        marks=pytest.mark.notyet(["duckdb", "flink", "pyspark"]),
    ),
    "read_parquet": pytest.param(
        BaseBackend,
        "read_parquet",
        marks=pytest.mark.notyet(["duckdb", "flink"]),
    ),
    "to_parquet_dir": pytest.param(
        BaseBackend,
        "to_parquet_dir",
        marks=pytest.mark.notyet(["pyspark"]),
    ),
}

params = _scrape_methods(
    [
        BaseBackend,
        SQLBackend,
        CanCreateCatalog,
        CanCreateDatabase,
        CanListCatalog,
        CanListDatabase,
    ],
    marks,
)

params.extend(marks.values())


@pytest.mark.parametrize("base_cls, method", params)
def test_signatures(base_cls, method, backend_cls):
    if not hasattr(backend_cls, method):
        pytest.skip(f"Method {method} not present in {backend_cls}, skipping...")

    base_sig = inspect.signature(getattr(base_cls, method))
    backend_sig = inspect.signature(getattr(backend_cls, method))

    # Usage is compatible(implementation_signature, defined_interface_signature, ...)
    assert compatible(backend_sig, base_sig, check_annotations=False)
