from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.tests.base import BackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class TestConf(BackendTest):
    """Wires chDB into the shared backend test suite.

    chDB is an embedded (in-process) engine, so unlike the ClickHouse backend
    there is no service to connect to. Standard test data is loaded straight
    into the engine from the shared parquet fixtures via ClickHouse's native
    ``file()`` table function.
    """

    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = "s"
    supported_to_timestamp_units = {"s"}
    supports_floating_modulus = False
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
            con.raw_query(stmt.replace("\n", " "))

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        # One persistent on-disk database per worker so tables survive across
        # the several connections a test session opens (chDB permits multiple
        # connections only against the same path).
        return ibis.chdb.connect(str(tmpdir / f"chdb_{worker_id}"), **kw)


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    """Session-scoped chDB connection with the standard test dataset loaded."""
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture
def mem(tmp_path_factory, worker_id):
    """A fresh in-memory-ish chDB connection with no preloaded data."""
    path = tmp_path_factory.mktemp("chdb") / f"unit_{worker_id}"
    con = ibis.chdb.connect(str(path))
    yield con
    con.disconnect()
