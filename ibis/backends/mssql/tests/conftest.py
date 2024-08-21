from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.tests.base import ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

MSSQL_USER = os.environ.get("IBIS_TEST_MSSQL_USER", "sa")
MSSQL_PASS = os.environ.get("IBIS_TEST_MSSQL_PASSWORD", "1bis_Testing!")
MSSQL_HOST = os.environ.get("IBIS_TEST_MSSQL_HOST", "localhost")
MSSQL_PORT = int(os.environ.get("IBIS_TEST_MSSQL_PORT", 1433))
IBIS_TEST_MSSQL_DB = os.environ.get("IBIS_TEST_MSSQL_DATABASE", "ibis_testing")
MSSQL_PYODBC_DRIVER = os.environ.get("IBIS_TEST_MSSQL_PYODBC_DRIVER", "FreeTDS")


class TestConf(ServiceBackendTest):
    # MSSQL has the same rounding behavior as postgres
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    supports_structs = False
    supports_json = False
    rounding_method = "half_to_even"
    service_name = "mssql"
    deps = ("pyodbc",)

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def postload(self, **kw: Any):
        self.connection = self.connect(database=IBIS_TEST_MSSQL_DB, **kw)

    def _load_data(self, *, database: str = IBIS_TEST_MSSQL_DB, **_):
        with self.connection._safe_raw_sql(
            "IF DB_ID('ibis_testing') is NULL BEGIN CREATE DATABASE [ibis_testing] END"
        ):
            pass

        super()._load_data(database=database, **_)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.mssql.connect(
            host=MSSQL_HOST,
            user=MSSQL_USER,
            password=MSSQL_PASS,
            port=MSSQL_PORT,
            driver=MSSQL_PYODBC_DRIVER,
            autocommit=True,
            **kw,
        )


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
