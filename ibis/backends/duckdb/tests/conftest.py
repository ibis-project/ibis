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
    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

        script_dir = data_directory.parent

        schema = (script_dir / 'schema' / 'duckdb.sql').read_text()

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
        con = TestConf(data_directory=data_dir)
        if not SANDBOXED:
            con.connection._load_extensions(
                ["httpfs", "postgres_scanner", "sqlite_scanner"]
            )
        return con

    @staticmethod
    def connect(data_directory: Path) -> BaseBackend:
        pytest.importorskip("duckdb")
        return ibis.duckdb.connect()  # type: ignore
