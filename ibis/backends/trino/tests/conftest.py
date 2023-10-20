from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest
import sqlglot as sg

import ibis
import ibis.expr.datatypes as dt
import ibis.selectors as s
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

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


class TestConf(BackendTest, RoundAwayFromZero):
    # trino rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = "s"
    supports_structs = True
    supports_map = True
    supports_tpch = True
    deps = ("sqlalchemy", "trino.sqlalchemy")

    _tpch_data_schema = "tpch.tiny"
    _tpch_query_schema = "hive.ibis_sf1"

    def preload(self):
        # create a minio host named trino
        subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "minio",
                "bash",
                "-c",
                "mc config host add trino http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD",
            ],
            check=True,
        )

        for path in self.test_files:
            directory = path.with_suffix("").name
            raw_data_path = f"/opt/data/raw/{path.name}"
            # copy from local to minio container
            subprocess.run(
                ["docker", "compose", "cp", str(path), f"minio:{raw_data_path}"],
                check=True,
            )
            # copy from minio container to trino minio host
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "exec",
                    "minio",
                    "mc",
                    "cp",
                    raw_data_path,
                    f"trino/warehouse/{directory}/{path.name}",
                ],
                check=True,
            )

    def _transform_tpch_sql(self, parsed):
        def add_catalog_and_schema(node):
            if isinstance(node, sg.exp.Table):
                catalog, db = self._tpch_query_schema.split(".")
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
        query_schema = self._tpch_query_schema
        data_schema = self._tpch_data_schema
        database, schema = query_schema.split(".")

        tables = con.list_tables(schema=self._tpch_data_schema)
        con.create_schema(schema, database=database, force=True)

        prefixes = {"partsupp": "ps"}

        # this is the type duckdb uses for numeric columns in TPC-H data
        decimal_type = dt.Decimal(15, 2)

        with con.begin() as c:
            for table in tables:
                prefix = prefixes.get(table, table[0])

                t = (
                    con.table(table, schema=data_schema).rename(f"{prefix}_{{}}".format)
                    # https://github.com/trinodb/trino/issues/19477
                    .mutate(
                        s.across(s.of_type(dt.float64), lambda c: c.cast(decimal_type))
                    )
                )

                sql = ibis.to_sql(t, dialect="trino")
                c.exec_driver_sql(
                    f"CREATE OR REPLACE VIEW {query_schema}.{table} AS {sql}"
                )

    def _tpch_table(self, name: str):
        return self.connection.table(
            self.default_identifier_case_fn(name),
            schema=self._tpch_query_schema,
        )

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
