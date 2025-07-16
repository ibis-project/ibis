from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Literal

import pytest
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.expr.schema as sch
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import ServiceBackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


TRINO_USER = os.environ.get(
    "IBIS_TEST_TRINO_USER", os.environ.get("TRINO_USER", "user")
)
TRINO_PASS = os.environ.get(
    "IBIS_TEST_TRINO_PASSWORD", os.environ.get("TRINO_PASSWORD", "")
)
TRINO_HOST = os.environ.get(
    "IBIS_TEST_TRINO_HOST", os.environ.get("TRINO_HOST", "localhost")
)
TRINO_PORT = int(
    os.environ.get("IBIS_TEST_TRINO_PORT", os.environ.get("TRINO_PORT", "8080"))
)


class TestConf(ServiceBackendTest):
    # trino rounds half to even for double precision and half away from zero
    # for numeric and decimal

    service_name = "minio"
    data_volume = "/bitnami/minio/data"
    returned_timestamp_unit = "s"
    supports_structs = True
    supports_map = True
    supports_tpch = True
    supports_tpcds = True
    deps = ("trino",)

    def preload(self):
        # copy files to the minio host
        super().preload()

        for suite in ["tpch", "tpcds"]:
            for path in self.data_dir.joinpath(suite).rglob("*.parquet"):
                subprocess.run(
                    [
                        "docker",
                        "compose",
                        "cp",
                        str(path),
                        f"{self.service_name}:{self.data_volume}/{suite}_{path.name}",
                    ],
                    check=False,
                )

                dirname = path.with_suffix("").name
                subprocess.run(
                    [
                        "docker",
                        "compose",
                        "exec",
                        self.service_name,
                        "mc",
                        "cp",
                        f"{self.data_volume}/{suite}_{path.name}",
                        f"data/trino/{suite}/{dirname}/",
                    ],
                    check=True,
                )

        for path in self.test_files:
            # minio doesn't allow underscores in bucket names
            dirname = path.with_suffix("").name
            # copy from minio container to trino minio host
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "exec",
                    self.service_name,
                    "mc",
                    "cp",
                    f"{self.data_volume}/{path.name}",
                    f"data/trino/{dirname}/",
                ],
                check=True,
            )

    def _tpc_table(self, name: str, benchmark: Literal["h", "ds"]):
        return self.connection.table(name, database=f"hive.tpc{benchmark}")

    def _transform_tpc_sql(self, parsed, *, suite, leaves):
        def add_catalog_and_schema(node):
            if isinstance(node, sg.exp.Table) and node.name in leaves:
                catalog = "hive"
                db = f"tpc{suite}"
                return node.__class__(
                    db=db,
                    catalog=catalog,
                    **{
                        k: v for k, v in node.args.items() if k not in ("db", "catalog")
                    },
                )
            return node

        result = parsed.transform(add_catalog_and_schema)
        return result

    def _load_tpc(self, *, suite, **_) -> None:
        """Create views of data in the TPC-H catalog that ships with Trino.

        This method create relations that have column names prefixed with the
        first one (or two in the case of partsupp -> ps) character table name
        to match the DuckDB TPC-H query conventions.
        """
        suite_name = f"tpc{suite}"
        sqls = generate_tpc_tables(suite_name, data_dir=self.data_dir)
        with self.connection.begin() as con:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS hive.{suite_name}")
            for stmt in sqls:
                raw_sql = stmt.sql("trino", pretty=True)
                con.execute(raw_sql)

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("parquet").glob("*.parquet")

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):  # noqa: ARG004
        return ibis.trino.connect(
            host=TRINO_HOST,
            port=TRINO_PORT,
            user=TRINO_USER,
            auth=TRINO_PASS,
            database="memory",
            schema="default",
            **kw,
        )

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


def generate_tpc_tables(suite_name, *, data_dir):
    import pyarrow.parquet as pq

    tables = {
        path.with_suffix("").name: sch.from_pyarrow_schema(
            pq.read_metadata(path).schema.to_arrow_schema()
        )
        for path in (data_dir / suite_name).rglob("*.parquet")
    }
    return (
        sge.Create(
            kind="TABLE",
            exists=True,
            this=sge.Schema(
                this=sg.table(name, db=suite_name, catalog="hive", quoted=True),
                expressions=schema.to_sqlglot_column_defs("trino"),
            ),
            properties=sge.Properties(
                expressions=[
                    sge.Property(
                        this="external_location",
                        value=sge.convert(f"s3a://trino/{suite_name}/{name}"),
                    ),
                    sge.Property(this="format", value=sge.convert("PARQUET")),
                ]
            ),
        )
        for name, schema in tables.items()
    )
