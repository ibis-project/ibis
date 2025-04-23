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

SKIP_METHODS = [
    "do_connect",
    "from_connection",
    # lots of backend-specific options in this method
    "create_table",
]


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
    "compile": pytest.param(BaseBackend, "compile"),
    "create_database": pytest.param(
        CanCreateDatabase, "create_database", marks=pytest.mark.notyet(["mysql"])
    ),
    "drop_database": pytest.param(
        CanCreateDatabase, "drop_database", marks=pytest.mark.notyet(["mysql"])
    ),
    "drop_table": pytest.param(
        SQLBackend, "drop_table", marks=pytest.mark.notyet(["druid"])
    ),
    "execute": pytest.param(SQLBackend, "execute"),
    "create_view": pytest.param(SQLBackend, "create_view"),
    "drop_view": pytest.param(SQLBackend, "drop_view"),
    "insert": pytest.param(SQLBackend, "insert", marks=pytest.mark.notyet(["impala"])),
    "list_databases": pytest.param(CanCreateDatabase, "list_databases"),
    "list_tables": pytest.param(BaseBackend, "list_tables"),
    "read_csv": pytest.param(
        BaseBackend,
        "read_csv",
        marks=pytest.mark.notyet(["duckdb", "pyspark", "datafusion"]),
    ),
    "read_delta": pytest.param(BaseBackend, "read_delta"),
    "read_json": pytest.param(
        BaseBackend, "read_json", marks=pytest.mark.notyet(["duckdb", "pyspark"])
    ),
    "read_parquet": pytest.param(
        BaseBackend, "read_parquet", marks=pytest.mark.notyet(["duckdb"])
    ),
    "to_parquet_dir": pytest.param(
        BaseBackend, "to_parquet_dir", marks=pytest.mark.notyet(["pyspark"])
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
