from __future__ import annotations

import concurrent.futures
from os import environ as env
from typing import TYPE_CHECKING, Any

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
        volume_prefix = "/Volumes/ibis_testing/default/testing_data/parquet"
        with (
            concurrent.futures.ThreadPoolExecutor() as exe,
            databricks.sql.connect(
                server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
                http_path=env["DATABRICKS_HTTP_PATH"],
                access_token=env["DATABRICKS_TOKEN"],
                staging_allowed_local_path=str(self.data_dir),
            ) as con,
        ):
            for fut in concurrent.futures.as_completed(
                exe.submit(
                    put_into,
                    con,
                    f"PUT '{file}' INTO '{volume_prefix}/{file.name}' OVERWRITE",
                )
                for file in files
            ):
                fut.result()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:
        return ibis.databricks.connect(
            server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
            http_path=env["DATABRICKS_HTTP_PATH"],
            access_token=env["DATABRICKS_TOKEN"],
            catalog="ibis_testing",
            schema="default",
            **kw,
        )
