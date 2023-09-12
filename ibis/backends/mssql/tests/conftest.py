from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.conftest import init_database
from ibis.backends.tests.base import RoundHalfToEven, ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

MSSQL_USER = os.environ.get("IBIS_TEST_MSSQL_USER", "sa")
MSSQL_PASS = os.environ.get("IBIS_TEST_MSSQL_PASSWORD", "1bis_Testing!")
MSSQL_HOST = os.environ.get("IBIS_TEST_MSSQL_HOST", "localhost")
MSSQL_PORT = int(os.environ.get("IBIS_TEST_MSSQL_PORT", 1433))
IBIS_TEST_MSSQL_DB = os.environ.get("IBIS_TEST_MSSQL_DATABASE", "ibis_testing")


class TestConf(ServiceBackendTest, RoundHalfToEven):
    # MSSQL has the same rounding behavior as postgres
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_structs = False
    supports_json = False
    service_name = "mssql"
    deps = "pymssql", "sqlalchemy"

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(
        self,
        *,
        user: str = MSSQL_USER,
        password: str = MSSQL_PASS,
        host: str = MSSQL_HOST,
        port: int = MSSQL_PORT,
        database: str = IBIS_TEST_MSSQL_DB,
        **_: Any,
    ) -> None:
        """Load test data into a MSSQL backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        init_database(
            url=sa.engine.make_url(
                f"mssql+pymssql://{user}:{password}@{host}:{port:d}/{database}"
            ),
            database=database,
            schema=self.ddl_script,
            isolation_level="AUTOCOMMIT",
            recreate=False,
        )

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.mssql.connect(
            host=MSSQL_HOST,
            user=MSSQL_USER,
            password=MSSQL_PASS,
            database=IBIS_TEST_MSSQL_DB,
            port=MSSQL_PORT,
            **kw,
        )


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
