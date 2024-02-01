from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.tests.base import ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

PG_USER = os.environ.get("IBIS_TEST_RISINGWAVE_USER", os.environ.get("PGUSER", "root"))
PG_PASS = os.environ.get(
    "IBIS_TEST_RISINGWAVE_PASSWORD", os.environ.get("PGPASSWORD", "")
)
PG_HOST = os.environ.get(
    "IBIS_TEST_RISINGWAVE_HOST", os.environ.get("PGHOST", "localhost")
)
PG_PORT = os.environ.get("IBIS_TEST_RISINGWAVE_PORT", os.environ.get("PGPORT", 4566))
IBIS_TEST_RISINGWAVE_DB = os.environ.get(
    "IBIS_TEST_RISINGWAVE_DATABASE", os.environ.get("PGDATABASE", "dev")
)


class TestConf(ServiceBackendTest):
    # postgres rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = "s"
    supports_structs = False
    rounding_method = "half_to_even"
    service_name = "risingwave"
    deps = ("psycopg2",)

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(self, **_: Any) -> None:
        """Load test data into a PostgreSQL backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        with self.connection._safe_raw_sql(";".join(self.ddl_script)):
            pass

    @staticmethod
    def connect(*, tmpdir, worker_id, port: int | None = None, **kw):
        con = ibis.risingwave.connect(
            host=PG_HOST,
            port=port or PG_PORT,
            user=PG_USER,
            password=PG_PASS,
            database=IBIS_TEST_RISINGWAVE_DB,
            **kw,
        )
        cursor = con.raw_sql("SET RW_IMPLICIT_FLUSH TO true;")
        cursor.close()
        return con


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="module")
def alltypes(con):
    return con.tables.functional_alltypes


@pytest.fixture(scope="module")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="module")
def intervals(con):
    return con.table("intervals")
