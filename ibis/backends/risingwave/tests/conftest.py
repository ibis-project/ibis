from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.conftest import init_database
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
    deps = "psycopg2", "sqlalchemy"

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(
        self,
        *,
        user: str = PG_USER,
        password: str = PG_PASS,
        host: str = PG_HOST,
        port: int = PG_PORT,
        database: str = IBIS_TEST_RISINGWAVE_DB,
        **_: Any,
    ) -> None:
        """Load test data into a Risingwave backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        init_database(
            url=sa.engine.make_url(
                f"risingwave://{user}:{password}@{host}:{port:d}/{database}"
            ),
            database=database,
            schema=self.ddl_script,
            isolation_level="AUTOCOMMIT",
            recreate=False,
        )

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
def db(con):
    return con.database()


@pytest.fixture(scope="module")
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope="module")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="module")
def alltypes_sqla(con, alltypes):
    name = alltypes.op().name
    return con._get_sqla_table(name)


@pytest.fixture(scope="module")
def intervals(con):
    return con.table("intervals")


@pytest.fixture
def translate():
    from ibis.backends.risingwave import Backend

    context = Backend.compiler.make_context()
    return lambda expr: Backend.compiler.translator_class(expr, context).get_result()
