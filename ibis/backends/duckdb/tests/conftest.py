from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend


class TestConf(BackendTest, RoundAwayFromZero):
    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

    @staticmethod
    def _load_data(
        data_dir,
        script_dir,
        database: str = "ibis_testing",
        **_: Any,
    ) -> None:
        """Load test data into a DuckDB backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        duckdb = pytest.importorskip("duckdb")

        schema = (script_dir / 'schema' / 'duckdb.sql').read_text()

        conn = duckdb.connect(str(data_dir / f"{database}.ddb"))
        for stmt in filter(None, map(str.strip, schema.split(';'))):
            conn.execute(stmt)

        for table in TEST_TABLES:
            src = data_dir / f'{table}.csv'
            conn.execute(
                f"COPY {table} FROM {str(src)!r} (DELIMITER ',', HEADER, SAMPLE_SIZE 1)"
            )

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def connect(data_directory: Path) -> BaseBackend:
        path = data_directory / "ibis_testing.ddb"
        return ibis.duckdb.connect(str(path))  # type: ignore
