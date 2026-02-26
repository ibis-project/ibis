from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

import ibis

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckdbBackend
    from ibis.backends.postgres import Backend as PGBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("DUCKDB_POSTGRES") is None, reason="avoiding CI shenanigans"
)
# we don't run any of these tests in CI, only locally, to avoid bringing a postgres.
# To run locally set env variable to True and once a postgres container is up run the test.


@pytest.fixture(scope="session")
def pgcon(tmp_path_factory, data_dir, worker_id):
    from ibis.backends.postgres.tests.conftest import TestConf as PGTestConf

    with PGTestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        conn: PGBackend = be.connection
        yield conn


@pytest.fixture
def pgurl(pgcon: PGBackend):
    i = pgcon.con.info
    return f"postgres://{i.user}:{i.password}@{i.host}:{i.port}/{i.dbname}"


def test_read_postgres(con: DuckdbBackend, pgurl: str):
    table = con.read_postgres(pgurl, table_name="functional_alltypes")
    assert table.count().execute()


def test_postgres_geometry(pgurl: str):
    # https://github.com/ibis-project/ibis/issues/11585
    attach_name = ibis.util.gen_name("pgdb")
    con: DuckdbBackend = ibis.duckdb.connect()
    con.load_extension("spatial")
    con.raw_sql(f"ATTACH DATABASE '{pgurl}' AS {attach_name} (TYPE POSTGRES)")

    pg_table = con.table("geo", database=(attach_name, "public"))
    assert any(pg_table[col].type().is_geospatial() for col in pg_table.columns)

    ddb_table = con.create_table("duckdb_geo", pg_table)
    assert any(ddb_table[col].type().is_geospatial() for col in ddb_table.columns)
