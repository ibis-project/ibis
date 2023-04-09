from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.conftest import SANDBOXED, TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend


class TestConf(BackendTest, RoundAwayFromZero):
    supports_map = True

    def __init__(self, data_directory: Path, **kwargs: Any) -> None:
        self.connection = self.connect(data_directory, **kwargs)

        script_dir = data_directory.parent

        schema = (script_dir / 'schema' / 'duckdb.sql').read_text()

        if not SANDBOXED:
            self.connection._load_extensions(
                ["httpfs", "postgres_scanner", "sqlite_scanner"]
            )

        with self.connection.begin() as con:
            for stmt in filter(None, map(str.strip, schema.split(';'))):
                con.exec_driver_sql(stmt)

            for table in TEST_TABLES:
                src = data_directory / f'{table}.csv'
                con.exec_driver_sql(
                    f"COPY {table} FROM {str(src)!r} (DELIMITER ',', HEADER, SAMPLE_SIZE 1)"
                )

    @staticmethod
    def _load_data(data_dir, script_dir, **_: Any) -> None:
        """Load test data into a DuckDB backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        """
        return TestConf(data_directory=data_dir)

    @staticmethod
    def connect(data_directory: Path, **kwargs: Any) -> BaseBackend:
        pytest.importorskip("duckdb")
        return ibis.duckdb.connect(**kwargs)  # type: ignore


@pytest.fixture
def con(data_directory, tmp_path: Path):
    return TestConf(data_directory, extension_directory=str(tmp_path)).connection
