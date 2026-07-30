from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.tests.base import BackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class TestConf(BackendTest):
    """Wires chDB into the shared backend test suite. Embedded engine (no
    service); test data is loaded from the shared parquet fixtures via
    ClickHouse's ``file()`` table function."""

    check_dtype = False
    returned_timestamp_unit = "s"
    supports_json = False
    force_sort = True
    rounding_method = "half_to_even"
    stateful = False
    deps = ("chdb",)

    @property
    def native_bool(self) -> bool:
        return True

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("parquet").glob("*.parquet")

    @property
    def ddl_script(self) -> Iterable[str]:
        parquet_dir = self.data_dir / "parquet"
        for stmt in super().ddl_script:
            yield stmt.format(parquet_dir=parquet_dir)

    def _load_data(self, *, database: str = "ibis_testing", **_: Any) -> None:
        con = self.connection.con
        con.raw_query(f"CREATE DATABASE IF NOT EXISTS {database} ENGINE = Atomic")
        con.raw_query(f"USE {database}")
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
    """The single on-disk path every connection in this process must share.

    chDB's embedded engine is a process-global singleton: while one connection
    is open, connecting to a *different* path raises BAD_ARGUMENTS. Under the
    randomized test order this means every fixture and test has to funnel
    through one path, so they all derive it from here (the same value
    ``TestConf.connect`` uses for the shared-suite connection).
    """
    return str(tmp_path_factory.getbasetemp() / f"chdb_{worker_id}")


@pytest.fixture
def mem(chdb_path):
    """A no-preloaded-data connection on the shared process path.

    Unit tests work in the ``default`` database; the shared dataset lives in
    ``ibis_testing``, so the two don't collide.
    """
    con = ibis.chdb.connect(chdb_path)
    yield con
    con.disconnect()
