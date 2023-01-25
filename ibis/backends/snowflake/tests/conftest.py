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
from ibis.util import consume

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend


def copy_into(con, data_dir: Path, table: str) -> None:
    stage = "ibis_testing"
    csv = f"{table}.csv"
    con.exec_driver_sql(
        f"PUT file://{data_dir.joinpath(csv).absolute()} @{stage}/{csv}"
    )
    con.exec_driver_sql(
        f"COPY INTO {table} FROM @{stage}/{csv} FILE_FORMAT = (FORMAT_NAME = ibis_testing)"
    )


class TestConf(BackendTest, RoundAwayFromZero):
    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

    @staticmethod
    def _load_data(
        data_dir, script_dir, database: str = "ibis_testing", **_: Any
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

        if (snowflake_url := os.environ.get("SNOWFLAKE_URL")) is None:
            pytest.skip("SNOWFLAKE_URL environment variable is not defined")

        url = sa.engine.make_url(snowflake_url).set(database="")
        con = sa.create_engine(url)

        dbschema = f"ibis_testing.{url.username}"

        stmts = [
            "CREATE DATABASE IF NOT EXISTS ibis_testing",
            f"CREATE SCHEMA IF NOT EXISTS {dbschema}",
            f"USE SCHEMA {dbschema}",
            *script_dir.joinpath("schema", "snowflake.sql").read_text().split(";"),
        ]

        with con.begin() as c:
            consume(map(c.exec_driver_sql, filter(None, map(str.strip, stmts))))

            # not much we can do to make this faster, but running these in
            # multiple threads seems to save about 2x
            with concurrent.futures.ThreadPoolExecutor() as exe:
                for result in concurrent.futures.as_completed(
                    map(
                        partial(exe.submit, partial(copy_into, c, data_dir)),
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
