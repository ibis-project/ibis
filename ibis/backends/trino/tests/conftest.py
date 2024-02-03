from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.expr.datatypes as dt
import ibis.selectors as s
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
    os.environ.get("IBIS_TEST_TRINO_PORT", os.environ.get("TRINO_PORT", 8080))
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
    deps = ("trino",)

    def preload(self):
        # copy files to the minio host
        super().preload()

        for path in self.test_files:
            # minio doesn't allow underscores in bucket names
            dirname = path.with_suffix("").name.replace("_", "-")
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
                    f"data/trino/{dirname}/{path.name}",
                ],
                check=True,
            )

    def _transform_tpch_sql(self, parsed):
        def add_catalog_and_schema(node):
            if isinstance(node, sg.exp.Table):
                catalog = "hive"
                db = "ibis_sf1"
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

    def load_tpch(self) -> None:
        """Create views of data in the TPC-H catalog that ships with Trino.

        This method create relations that have column names prefixed with the
        first one (or two in the case of partsupp -> ps) character table name
        to match the DuckDB TPC-H query conventions.
        """
        con = self.connection
        database = "hive"
        schema = "ibis_sf1"

        tables = con.list_tables(schema="tiny", database="tpch")
        con.create_schema(schema, database=database, force=True)

        prefixes = {"partsupp": "ps"}

        # this is the type duckdb uses for numeric columns in TPC-H data
        decimal_type = dt.Decimal(15, 2)

        with con.begin() as c:
            for table in tables:
                prefix = prefixes.get(table, table[0])

                t = (
                    con.table(table, schema="tiny", database="tpch")
                    .rename(f"{prefix}_{{}}".format)
                    # https://github.com/trinodb/trino/issues/19477
                    .mutate(
                        s.across(s.of_type(dt.float64), lambda c: c.cast(decimal_type))
                    )
                )

                sql = sge.Create(
                    kind="VIEW",
                    this=sg.table(table, db=schema, catalog=database),
                    expression=self.connection._to_sqlglot(t),
                    replace=True,
                ).sql("trino", pretty=True)

                c.execute(sql)

    def _tpch_table(self, name: str):
        from ibis import _

        table = self.connection.table(name, schema="ibis_sf1", database="hive")
        table = table.mutate(s.across(s.of_type("double"), _.cast("decimal(15, 2)")))
        return table

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("parquet").glob("*.parquet")

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.trino.connect(
            host=TRINO_HOST,
            port=TRINO_PORT,
            user=TRINO_USER,
            password=TRINO_PASS,
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
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="module")
def db(con):
    return con.database()


@pytest.fixture(scope="module")
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope="module")
def geotable(con):
    return con.table("geo")


@pytest.fixture(scope="module")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="module")
def gdf(geotable):
    return geotable.execute()


@pytest.fixture(scope="module")
def intervals(con):
    return con.table("intervals")


@pytest.fixture
def translate():
    from ibis.backends.trino import Backend

    context = Backend.compiler.make_context()
    return lambda expr: (Backend.compiler.translator_class(expr, context).get_result())
