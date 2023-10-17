from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.conftest import SANDBOXED

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ibis.backends.base import BaseBackend


class TestConf(BackendTest, RoundAwayFromZero):
    supports_map = True
    deps = "duckdb", "duckdb_engine"
    stateful = False
    supports_tpch = True

    def preload(self):
        if not SANDBOXED:
            self.connection._load_extensions(
                ["httpfs", "postgres_scanner", "sqlite_scanner"]
            )

    @property
    def ddl_script(self) -> Iterator[str]:
        parquet_dir = self.data_dir / "parquet"
        for table in TEST_TABLES:
            yield (
                f"""
                CREATE OR REPLACE TABLE {table} AS
                SELECT * FROM read_parquet('{parquet_dir / f'{table}.parquet'}')
                """
            )
        yield from super().ddl_script

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:
        # use an extension directory per test worker to prevent simultaneous
        # downloads
        extension_directory = tmpdir.getbasetemp().joinpath("duckdb_extensions")
        extension_directory.mkdir(exist_ok=True)
        return ibis.duckdb.connect(extension_directory=extension_directory, **kw)

    def load_tpch(self) -> None:
        with self.connection.begin() as con:
            con.exec_driver_sql("CALL dbgen(sf=0.1)")


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
