from __future__ import annotations

import concurrent.futures
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
    deps = ("snowflake.connector", "snowflake.sqlalchemy")

    def _load_data(self, **_: Any) -> None:
        """Load test data into a Snowflake backend instance."""
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
{self.script_dir.joinpath("snowflake.sql").read_text()}"""
            )

        with con.begin() as c:
            # not much we can do to make this faster, but running these in
            # multiple threads seems to save about 2x
            with concurrent.futures.ThreadPoolExecutor() as exe:
                for future in concurrent.futures.as_completed(
                    exe.submit(copy_into, c, self.data_dir, table)
                    for table in TEST_TABLES.keys()
                ):
                    future.result()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:
        return ibis.connect(_get_url(), **kw)


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
