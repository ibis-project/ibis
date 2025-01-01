from __future__ import annotations

import concurrent.futures
import getpass
import sys
from os import environ as env
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest

if TYPE_CHECKING:
    import s3fs

    from ibis.backends import BaseBackend


IBIS_ATHENA_S3_STAGING_DIR = env.get(
    "IBIS_ATHENA_S3_STAGING_DIR", "s3://aws-athena-query-results-ibis-testing/"
)
AWS_REGION = env.get("AWS_REGION", "us-east-2")
AWS_PROFILE = env.get("AWS_PROFILE")
CONNECT_ARGS = dict(
    s3_staging_dir=IBIS_ATHENA_S3_STAGING_DIR,
    region_name=AWS_REGION,
    profile_name=AWS_PROFILE,
)


def create_table(con, *, fs: s3fs.S3FileSystem, file: str, folder: str) -> None:
    import pyarrow.parquet as pq

    from ibis.formats.pyarrow import PyArrowSchema

    arrow_schema = pq.read_metadata(file).schema.to_arrow_schema()
    schema = PyArrowSchema.to_ibis(arrow_schema).to_sqlglot("athena")
    name = file.with_suffix("").name

    ddl = sge.Create(
        kind="TABLE",
        this=sge.Schema(this=sg.table(name), expressions=schema),
        properties=sge.Properties(
            expressions=[
                sge.ExternalProperty(),
                sge.FileFormatProperty(this=sge.Var(this="PARQUET")),
                sge.LocationProperty(this=sge.convert(f"{folder}/{name}")),
            ]
        ),
    )

    fs.put(str(file), f"{folder.removeprefix('s3://')}/{name}/{file.name}")

    drop_query = sge.Drop(kind="TABLE", this=sg.table(name), exists=True).sql("athena")
    create_query = ddl.sql("athena")

    with con.cursor() as cur:
        cur.execute(drop_query)
        cur.execute(create_query)


class TestConf(BackendTest):
    supports_map = False
    supports_json = False
    supports_structs = False
    driver_supports_multiple_statements = False
    deps = ("pyathena", "s3fs")

    def _load_data(self, **_: Any) -> None:
        import pyathena
        import s3fs

        files = list(self.data_dir.joinpath("parquet").glob("*.parquet"))

        user = getpass.getuser()
        python_version = "".join(map(str, sys.version_info[:3]))
        folder = f"{user}_{python_version}"

        fs = s3fs.S3FileSystem()

        futures = []

        with (
            pyathena.connect(**CONNECT_ARGS) as con,
            concurrent.futures.ThreadPoolExecutor() as executor,
        ):
            for file in files:
                futures.append(
                    executor.submit(
                        create_table,
                        con,
                        fs=fs,
                        file=file,
                        folder=f"{IBIS_ATHENA_S3_STAGING_DIR}{folder}",
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                future.result()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:
        return ibis.athena.connect(**CONNECT_ARGS, **kw)

    def _remap_column_names(self, table_name: str) -> dict[str, str]:
        table = self.connection.table(table_name)
        return table.rename(
            dict(zip(TEST_TABLES[table_name].names, table.schema().names))
        )

    @property
    def batting(self):
        return self._remap_column_names("batting")

    @property
    def awards_players(self):
        return self._remap_column_names("awards_players")
