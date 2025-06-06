from __future__ import annotations

import concurrent.futures
import contextlib
import itertools
import os
import subprocess
from typing import TYPE_CHECKING, Any

import oracledb
import pytest

import ibis
from ibis.backends.tests.base import ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

ORACLE_USER = os.environ.get("IBIS_TEST_ORACLE_USER", "ibis")
ORACLE_PASS = os.environ.get("IBIS_TEST_ORACLE_PASSWORD", "ibis")
ORACLE_HOST = os.environ.get("IBIS_TEST_ORACLE_HOST", "localhost")
ORACLE_PORT = int(os.environ.get("IBIS_TEST_ORACLE_PORT", "1521"))

# Creating test DB and user
# The ORACLE_DB env-var needs to be set in the compose.yaml file
# Then, after the container is running, exec in and run (from `/opt/oracle`)
# ./createAppUser user pass ORACLE_DB
# where ORACLE_DB is the same name you used in the Compose file.

# Set to ensure decimals come back as decimals
oracledb.defaults.fetch_decimals = True


class TestConf(ServiceBackendTest):
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    # Automatic mutate so that logical type in ibis returns as bool
    # but the backend will still do whatever it's going to do
    native_bool = False
    supports_structs = False
    supports_json = False
    rounding_method = "half_to_even"
    data_volume = "/opt/oracle/data"
    service_name = "oracle"
    deps = ("oracledb",)

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
                    self.service_name,
                    "./createAppUser",
                    user,
                    password,
                    database,
                ]
            )

        init_oracle_database(
            dsn=oracledb.makedsn(host, port, service_name=database),
            user=user,
            password=password,
            schema=self.ddl_script,
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
                        self.service_name,
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
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
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
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection


def init_oracle_database(
    user: str, password: str, dsn: str, schema: str | None = None
) -> None:
    with oracledb.connect(
        dsn, user=user, password=password, stmtcachesize=0
    ).cursor() as cursor:
        for stmt in schema:
            # XXX: maybe should just remove the comments in the sql file
            # so we don't end up writing an entire parser here.
            if not stmt.startswith("--"):
                cursor.execute(stmt)
