from __future__ import annotations

import concurrent.futures
import functools
import os
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from pathlib import Path

    from ibis.backends.base import BaseBackend


def _get_url():
    if (url := os.environ.get("SNOWFLAKE_URL")) is not None:
        return url
    else:
        try:
            user, password, account, database, schema, warehouse = tuple(
                os.environ[f"SNOWFLAKE_{part}"]
                for part in [
                    "USER",
                    "PASSWORD",
                    "ACCOUNT",
                    "DATABASE",
                    "SCHEMA",
                    "WAREHOUSE",
                ]
            )
        except KeyError as e:
            pytest.skip(
                f"missing URL part: {e} or SNOWFLAKE_URL environment variable is not defined"
            )
        return f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"


def copy_into(con, data_dir: Path, table: str) -> None:
    stage = "ibis_testing"
    csv = f"{table}.csv"
    con.exec_driver_sql(
        f"PUT file://{data_dir.joinpath('csv', csv).absolute()} @{stage}/{csv}"
    )
    con.exec_driver_sql(
        f"COPY INTO {table} FROM @{stage}/{csv} FILE_FORMAT = (FORMAT_NAME = ibis_testing)"
    )


class TestConf(BackendTest, RoundAwayFromZero):
    supports_map = True
    default_identifier_case_fn = staticmethod(str.upper)

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

        snowflake_url = _get_url()

        raw_url = sa.engine.make_url(snowflake_url)
        _, schema = raw_url.database.rsplit("/", 1)
        url = raw_url.set(database="")
        con = sa.create_engine(
            url, connect_args={"session_parameters": {"MULTI_STATEMENT_COUNT": "0"}}
        )

        dbschema = f"ibis_testing.{schema}"

        with con.begin() as c:
            c.exec_driver_sql(
                f"""\
CREATE DATABASE IF NOT EXISTS ibis_testing;
USE DATABASE ibis_testing;
CREATE SCHEMA IF NOT EXISTS {dbschema};
USE SCHEMA {dbschema};
{script_dir.joinpath("schema", "snowflake.sql").read_text()}"""
            )

        with con.begin() as c:
            # not much we can do to make this faster, but running these in
            # multiple threads seems to save about 2x
            with concurrent.futures.ThreadPoolExecutor() as exe:
                for future in concurrent.futures.as_completed(
                    exe.submit(copy_into, c, data_dir, table)
                    for table in TEST_TABLES.keys()
                ):
                    future.result()

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def connect(data_directory: Path) -> BaseBackend:
        return ibis.connect(_get_url())


@pytest.fixture(scope="session")
def con(data_directory):
    return TestConf.connect(data_directory)
