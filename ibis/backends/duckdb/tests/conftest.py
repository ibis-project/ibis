from functools import lru_cache
from pathlib import Path

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

    @staticmethod
    def _load_data(data_dir, script_dir, **kwargs):
        """Load testdata into an impala backend.

        Parameters
        ----------
        data_dir : Path
            Location of testdata
        script_dir : Path
            Location of scripts defining schemas
        """

        database = kwargs.get("database", "ibis_testing")

        schema = (script_dir / 'schema' / 'duckdb.sql').read_text()
        tables = TEST_TABLES

        import duckdb  # noqa: F401

        conn = duckdb.connect(str(data_dir / f"{database}.ddb"))
        for stmt in filter(None, map(str.strip, schema.split(';'))):
            conn.execute(stmt)

        for table in tables:
            src = data_dir / f'{table}.csv'
            conn.execute(
                f"COPY {table} FROM '{src}'"
                " (DELIMITER ',', HEADER, SAMPLE_SIZE 1)"
            )

    @staticmethod
    @lru_cache(maxsize=None)
    def connect(data_directory: Path):
        path = data_directory / "ibis_testing.ddb"
        return ibis.duckdb.connect(str(path))  # type: ignore
