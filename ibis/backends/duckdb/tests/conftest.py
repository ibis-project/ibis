from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import pytest

import ibis
from ibis.backends.conftest import SANDBOXED, TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend


class TestConf(BackendTest, RoundAwayFromZero):
    supports_map = True
    deps = "duckdb", "duckdb_engine"
    stateful = False

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
        # extension directory per test worker to prevent simultaneous downloads
        return ibis.duckdb.connect(
            extension_directory=str(tmpdir.mktemp(f"{worker_id}_exts")), **kw
        )


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
