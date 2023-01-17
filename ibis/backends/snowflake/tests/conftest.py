from __future__ import annotations

import concurrent.futures
import functools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend


def copy_into(con, data_dir: Path, table: str) -> None:
    stage = "ibis_testing_stage"
    csv = f"{table}.csv"
    src = data_dir / csv
    con.execute(sa.text(f"PUT file://{src.absolute()} @{stage}/{csv}"))
    con.execute(
        sa.text(
            f"COPY INTO {table} FROM @{stage}/{csv} FILE_FORMAT = (FORMAT_NAME = ibis_csv_fmt)"
        )
    )


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
        """Load test data into a Snowflake backend instance.

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

        con = TestConf.connect(data_dir)

        with con.begin() as con:
            con.execute(sa.text("USE WAREHOUSE ibis_testing"))
            con.execute(sa.text(f"DROP DATABASE IF EXISTS {database}"))
            con.execute(sa.text(f"CREATE DATABASE IF NOT EXISTS {database}"))
            con.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {database}.ibis_testing"))
            con.execute(sa.text(f"USE SCHEMA {database}.ibis_testing"))

            for stmt in filter(None, map(str.strip, schema.split(';'))):
                con.execute(sa.text(stmt))

            # not much we can do to make this faster, but running these in
            # multiple threads seems to save about 2x
            with concurrent.futures.ThreadPoolExecutor() as exe:
                for result in concurrent.futures.as_completed(
                    map(
                        partial(exe.submit, partial(copy_into, con, data_dir)),
                        TEST_TABLES,
                    )
                ):
                    result.result()

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def connect(data_directory: Path) -> BaseBackend:
        if snowflake_url := os.environ.get("SNOWFLAKE_URL"):
            return ibis.connect(snowflake_url)  # type: ignore
        pytest.skip("SNOWFLAKE_URL environment variable is not defined")  # noqa: RET503
