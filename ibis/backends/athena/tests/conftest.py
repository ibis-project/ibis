from __future__ import annotations

import concurrent.futures
import getpass
import sys
from os import environ as env
from typing import TYPE_CHECKING, Any

import pytest
import sqlglot as sg
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
IBIS_ATHENA_TEST_DATABASE = (
    f"{getpass.getuser()}_{''.join(map(str, sys.version_info[:3]))}"
)
AWS_REGION = env.get("AWS_REGION", "us-east-2")
AWS_PROFILE = env.get("AWS_PROFILE")
CONNECT_ARGS = dict(
    s3_staging_dir=f"{IBIS_ATHENA_S3_STAGING_DIR}/",
    region_name=AWS_REGION,
    profile_name=AWS_PROFILE,
)


def create_table(connection, *, fs: s3fs.S3FileSystem, file: Path, folder: str) -> None:
    from ibis.formats.pyarrow import PyArrowSchema

    arrow_schema = pq.read_metadata(file).schema.to_arrow_schema()
    ibis_schema = PyArrowSchema.to_ibis(arrow_schema)
    name = file.with_suffix("").name

    fs.put(str(file), f"{folder.removeprefix('s3://')}/{name}/{file.name}")

    connection.drop_table(name, database=IBIS_ATHENA_TEST_DATABASE, force=True)

    t = connection.create_table(
        name,
        schema=ibis_schema,
        location=f"{folder}/{name}",
        database=IBIS_ATHENA_TEST_DATABASE,
    )

    assert t.count().execute() > 0


class TestConf(BackendTest):
    supports_map = False
    supports_json = False
    supports_structs = False

    driver_supports_multiple_statements = False

    deps = ("pyathena", "fsspec")

    @staticmethod
    def format_table(name: str) -> str:
        return sg.table(name, db=IBIS_ATHENA_TEST_DATABASE, quoted=True).sql(Athena)

    def _load_data(self, **_: Any) -> None:
        import fsspec

        files = self.data_dir.joinpath("parquet").glob("*.parquet")

        fs = fsspec.filesystem("s3")

        connection = self.connection
        db_dir = f"{IBIS_ATHENA_S3_STAGING_DIR}/{IBIS_ATHENA_TEST_DATABASE}"

        connection.create_database(
            IBIS_ATHENA_TEST_DATABASE, location=db_dir, force=True
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for future in concurrent.futures.as_completed(
                executor.submit(
                    create_table, connection, fs=fs, file=file, folder=db_dir
                )
                for file in files
            ):
                future.result()

    def postload(self, **kw):
        self.connection = self.connect(schema_name=IBIS_ATHENA_TEST_DATABASE, **kw)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:  # noqa: ARG004
        return ibis.athena.connect(**CONNECT_ARGS, **kw)

    def _remap_column_names(self, table_name: str) -> dict[str, str]:
        table = self.connection.table(table_name, database=IBIS_ATHENA_TEST_DATABASE)
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
