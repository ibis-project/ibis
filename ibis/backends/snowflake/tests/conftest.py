from __future__ import annotations

import functools
import os
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
        pytest.importorskip("snowflake.connector")
        pytest.importorskip("snowflake.sqlalchemy")

        schema = (script_dir / 'schema' / 'snowflake.sql').read_text()

        stage = "ibis_testing_stage"
        con = TestConf.connect(data_dir)
        with con.con.begin() as con:
            con.execute("DROP SCHEMA IF EXISTS ibis_testing")
            con.execute("CREATE SCHEMA IF NOT EXISTS ibis_testing")
            con.execute("USE SCHEMA ibis_testing")
            con.execute(
                """\
CREATE OR REPLACE FILE FORMAT ibis_csv_fmt
    type = 'CSV'
    field_delimiter = ','
    skip_header = 1
    field_optionally_enclosed_by = '"'"""
            )
            con.execute(
                """\
CREATE OR REPLACE STAGE ibis_testing_stage
    file_format = ibis_csv_fmt;"""
            )
            for stmt in filter(None, map(str.strip, schema.split(';'))):
                con.execute(stmt)

            for table in TEST_TABLES:
                src = data_dir / f"{table}.csv"
                con.execute(f"PUT file://{str(src.absolute())} @{stage}/{table}.csv")
                con.execute(
                    f"COPY INTO {table} FROM @{stage}/{table}.csv FILE_FORMAT = (FORMAT_NAME = ibis_csv_fmt)"
                )

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def connect(data_directory: Path) -> BaseBackend:
        if snowflake_url := os.environ.get("SNOWFLAKE_URL"):
            return ibis.connect(snowflake_url)  # type: ignore
        pytest.skip("SNOWFLAKE_URL environment variable is not defined")
        return None
