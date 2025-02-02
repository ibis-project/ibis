from __future__ import annotations

import concurrent.futures
import getpass
import sys
from os import environ as env
from typing import TYPE_CHECKING, Any

import pytest
import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot.dialects import Athena

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest

if TYPE_CHECKING:
    from pathlib import Path

    import s3fs

    from ibis.backends import BaseBackend


pq = pytest.importorskip("pyarrow.parquet")


IBIS_ATHENA_S3_STAGING_DIR = env.get(
    "IBIS_ATHENA_S3_STAGING_DIR", "s3://aws-athena-query-results-ibis-testing"
)
AWS_REGION = env.get("AWS_REGION", "us-east-2")
AWS_PROFILE = env.get("AWS_PROFILE")
CONNECT_ARGS = dict(
    s3_staging_dir=f"{IBIS_ATHENA_S3_STAGING_DIR}/",
    region_name=AWS_REGION,
    profile_name=AWS_PROFILE,
)


def create_table(con, *, fs: s3fs.S3FileSystem, file: Path, folder: str) -> None:
    from ibis.formats.pyarrow import PyArrowSchema

    arrow_schema = pq.read_metadata(file).schema.to_arrow_schema()
    ibis_schema = PyArrowSchema.to_ibis(arrow_schema)
    sg_schema = ibis_schema.to_sqlglot(Athena)
    name = file.with_suffix("").name

    ddl = sge.Create(
        kind="TABLE",
        exists=True,
        this=sge.Schema(this=sg.table(name), expressions=sg_schema),
        properties=sge.Properties(
            expressions=[
                sge.ExternalProperty(),
                sge.FileFormatProperty(this=sge.Var(this="PARQUET")),
                sge.LocationProperty(this=sge.convert(f"{folder}/{name}")),
            ]
        ),
    )

    fs.put(str(file), f"{folder.removeprefix('s3://')}/{name}/{file.name}")

    create_query = ddl.sql(Athena)

    with con.cursor() as cur:
        cur.execute(create_query)


class TestConf(BackendTest):
    supports_map = False
    supports_json = False
    supports_structs = False

    driver_supports_multiple_statements = False

    deps = ("pyathena", "fsspec")

    def _load_data(self, **_: Any) -> None:
        import fsspec

        files = self.data_dir.joinpath("parquet").glob("*.parquet")

        user = getpass.getuser()
        python_version = "".join(map(str, sys.version_info[:3]))
        folder = f"{user}_{python_version}"

        fs = fsspec.filesystem("s3")

        con = self.connection.con
        folder = f"{IBIS_ATHENA_S3_STAGING_DIR}/{folder}"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for future in concurrent.futures.as_completed(
                executor.submit(create_table, con, fs=fs, file=file, folder=folder)
                for file in files
            ):
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


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    with TestConf.load_data(data_dir, tmp_path_factory, worker_id) as be:
        yield be.connection
