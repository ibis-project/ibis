from __future__ import annotations

import concurrent.futures
import getpass
import sys
from os import environ as env
from typing import TYPE_CHECKING, Any

import pytest

import ibis
from ibis.backends.tests.base import BackendTest

if TYPE_CHECKING:
    from ibis.backends import BaseBackend


def put_into(con, query):
    with con.cursor() as cur:
        cur.execute(query)


class TestConf(BackendTest):
    supports_map = True
    driver_supports_multiple_statements = False
    deps = ("databricks.sql",)

    def _load_data(self, **_: Any) -> None:
        import databricks.sql

        files = list(self.data_dir.joinpath("parquet").glob("*.parquet"))

        user = getpass.getuser()
        python_version = "".join(map(str, sys.version_info[:3]))
        volume = f"{user}_{python_version}"
        volume_prefix = f"/Volumes/ibis_testing/default/{volume}"

        with databricks.sql.connect(
            server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
            http_path=env["DATABRICKS_HTTP_PATH"],
            access_token=env["DATABRICKS_TOKEN"],
            staging_allowed_local_path=str(self.data_dir),
        ) as con:
            with con.cursor() as cur:
                cur.execute(
                    f"CREATE VOLUME IF NOT EXISTS ibis_testing.default.{volume} COMMENT 'Ibis test data storage'"
                )
            with concurrent.futures.ThreadPoolExecutor() as exe:
                for fut in concurrent.futures.as_completed(
                    exe.submit(
                        put_into,
                        con,
                        f"PUT '{file}' INTO '{volume_prefix}/{file.name}' OVERWRITE",
                    )
                    for file in files
                ):
                    fut.result()

            with con.cursor() as cur:
                for raw_stmt in self.ddl_script:
                    try:
                        stmt = raw_stmt.format(user=user, python_version=python_version)
                    except KeyError:  # not a valid format string, just execute it
                        stmt = raw_stmt

                    cur.execute(stmt)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:  # noqa: ARG004
        return ibis.databricks.connect(
            server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
            http_path=env["DATABRICKS_HTTP_PATH"],
            access_token=env["DATABRICKS_TOKEN"],
            catalog="ibis_testing",
            schema="default",
            **kw,
        )


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
