from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis import util
from ibis.backends.tests.base import BackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class TestConf(BackendTest):
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_json = False
    force_sort = True
    rounding_method = "half_to_even"
    stateful = False
    deps = ("chdb",)

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("parquet").glob("*.parquet")

    def _load_data(self, *, database: str = "ibis_testing", **_: Any) -> None:
        con = self.connection.con
        con.raw_query(f"CREATE DATABASE IF NOT EXISTS {database} ENGINE = Atomic")
        con.raw_query(f"USE {database}")
        parquet = self.data_dir / "parquet"
        for name, select in (
            ("diamonds", "SELECT *"),
            ("batting", "SELECT *"),
            ("awards_players", "SELECT *"),
            # timestamp_col carries a tz that chDB must drop
            (
                "functional_alltypes",
                "SELECT * REPLACE(CAST(CAST(timestamp_col AS Nullable(String)) "
                "AS Nullable(DateTime)) AS timestamp_col)",
            ),
            ("astronauts", "SELECT *"),
        ):
            path = parquet / f"{name}.parquet"
            con.raw_query(
                f"CREATE OR REPLACE TABLE {name} ENGINE = Memory AS "
                f"{select} FROM file('{path}', 'Parquet')"
            )
        for stmt in self.ddl_script:
            if stmt.strip():
                con.raw_query(stmt)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        # Persistent on-disk db per worker (tmpdir is a TempPathFactory) so the
        # process's several connections share one path, as chDB requires.
        path = tmpdir.getbasetemp() / f"chdb_{worker_id}"
        return ibis.chdb.connect(str(path), **kw)


@pytest.fixture(scope="session")
def chdb_path(tmp_path_factory, worker_id) -> str:
    # chDB allows one engine path per process; every connection funnels
    # through this one.
    return str(tmp_path_factory.getbasetemp() / f"chdb_{worker_id}")


@pytest.fixture
def mem(chdb_path):
    # isolated per-test database on the shared engine path
    con = ibis.chdb.connect(chdb_path)
    name = f"ibis_unit_{util.guid()}"
    con.raw_sql(f"CREATE DATABASE {name} ENGINE = Atomic")
    con.con.raw_query(f"USE {name}")
    yield con
    con.raw_sql(f"DROP DATABASE IF EXISTS {name}")
    con.disconnect()
