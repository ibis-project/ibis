from __future__ import annotations

import concurrent.futures
import contextlib
import itertools
import os
import subprocess
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

import ibis
from ibis.backends.tests.base import RoundHalfToEven, ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

ORACLE_USER = os.environ.get("IBIS_TEST_ORACLE_USER", "ibis")
ORACLE_PASS = os.environ.get("IBIS_TEST_ORACLE_PASSWORD", "ibis")
ORACLE_HOST = os.environ.get("IBIS_TEST_ORACLE_HOST", "localhost")
ORACLE_PORT = int(os.environ.get("IBIS_TEST_ORACLE_PORT", 1521))

# Creating test DB and user
# The ORACLE_DB env-var needs to be set in the docker-compose.yml file
# Then, after the container is running, exec in and run (from `/opt/oracle`)
# ./createAppUser user pass ORACLE_DB
# where ORACLE_DB is the same name you used in the docker-compose file.


class TestConf(ServiceBackendTest, RoundHalfToEven):
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    supports_arrays_outside_of_select = False
    # Automatic mutate so that logical type in ibis returns as bool
    # but the backend will still do whatever it's going to do
    native_bool = False
    supports_structs = False
    supports_json = False
    data_volume = "/opt/oracle/data"
    service_name = "oracle"
    deps = "oracledb", "sqlalchemy"

    @property
    def test_files(self) -> Iterable[Path]:
        return itertools.chain(
            self.data_dir.joinpath("csv").glob("*.csv"),
            self.script_dir.joinpath("oracle").glob("*.ctl"),
        )

    def _load_data(
        self,
        *,
        user: str = ORACLE_USER,
        password: str = ORACLE_PASS,
        host: str = ORACLE_HOST,
        port: int = ORACLE_PORT,
        **_: Any,
    ) -> None:
        """Load test data into a Oracle backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        database = "IBIS_TESTING"

        # suppress the exception if the user/pass/db combo already exists
        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.check_call(
                [
                    "docker",
                    "compose",
                    "exec",
                    "oracle",
                    "./createAppUser",
                    user,
                    password,
                    database,
                ]
            )

        init_oracle_database(
            url=sa.engine.make_url(
                f"oracle://{user}:{password}@{host}:{port:d}/{database}",
            ),
            database=database,
            schema=self.ddl_script,
            connect_args=dict(service_name=database),
        )

        # then call sqlldr to ingest
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for fut in concurrent.futures.as_completed(
                executor.submit(
                    subprocess.check_call,
                    [
                        "docker",
                        "compose",
                        "exec",
                        "oracle",
                        "sqlldr",
                        f"{user}/{password}@{host}:{port:d}/{database}",
                        f"control=data/{ctl_file.name}",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                for ctl_file in self.script_dir.joinpath("oracle").glob("*.ctl")
            ):
                fut.result()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.oracle.connect(
            host=ORACLE_HOST,
            user=ORACLE_USER,
            password=ORACLE_PASS,
            database="IBIS_TESTING",
            port=ORACLE_PORT,
            **kw,
        )

    @staticmethod
    def format_table(name: str) -> str:
        return f'"{name}"'


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


def init_oracle_database(
    url: sa.engine.url.URL,
    database: str,
    schema: str | None = None,
    **kwargs: Any,
) -> sa.engine.Engine:
    """Initialise `database` at `url` with `schema`.

    If `recreate`, drop the `database` at `url`, if it exists.

    Parameters
    ----------
    url : url.sa.engine.url.URL
        Connection url to the database
    database : str
        Name of the database to be dropped
    schema : TextIO
        File object containing schema to use

    Returns
    -------
    sa.engine.Engine
        SQLAlchemy engine object
    """
    try:
        url.database = database
    except AttributeError:
        url = url.set(database=database)

    engine = sa.create_engine(url, **kwargs)

    if schema:
        with engine.begin() as conn:
            for stmt in schema:
                # XXX: maybe should just remove the comments in the sql file
                # so we don't end up writing an entire parser here.
                if not stmt.startswith("--"):
                    conn.exec_driver_sql(stmt)

    return engine
