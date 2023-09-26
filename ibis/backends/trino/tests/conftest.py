from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
import sqlglot as sg

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.postgres.tests.conftest import TestConf as PostgresTestConf
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.backends.tests.data import struct_types

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
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
TRINO_PORT = os.environ.get("IBIS_TEST_TRINO_PORT", os.environ.get("TRINO_PORT", 8080))
IBIS_TEST_TRINO_DB = os.environ.get(
    "IBIS_TEST_TRINO_DATABASE",
    os.environ.get("TRINO_DATABASE", "memory"),
)


class TrinoPostgresTestConf(PostgresTestConf):
    service_name = "trino-postgres"
    deps = "sqlalchemy", "psycopg2"

    @classmethod
    def name(cls) -> str:
        return "postgres"

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")


class TestConf(BackendTest, RoundAwayFromZero):
    # trino rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = "s"
    supports_structs = True
    supports_map = True
    supports_tpch = True
    service_name = "trino"
    deps = ("sqlalchemy", "trino.sqlalchemy")

    _tpch_data_schema = "tpch.sf1"
    _tpch_query_schema = "hive.ibis_sf1"

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

        tables = con.list_tables(database=self._tpch_data_schema)
        con.create_schema(schema, database=database, force=True)

        prefixes = {"partsupp": "ps"}
        with con.begin() as c:
            for table in tables:
                prefix = prefixes.get(table, table[0])

                t = con.table(table, schema=data_schema)
                new_t = t.rename(**{f"{prefix}_{old}": old for old in t.columns})

                sql = ibis.to_sql(new_t, dialect="trino")
                c.exec_driver_sql(
                    f"CREATE OR REPLACE VIEW {query_schema}.{table} AS {sql}"
                )

    def _tpch_table(self, name: str):
        return self.connection.table(
            self.default_identifier_case_fn(name),
            schema=self._tpch_query_schema,
        )

    @classmethod
    def load_data(cls, data_dir: Path, tmpdir: Path, worker_id: str, **kw: Any) -> None:
        TrinoPostgresTestConf.load_data(data_dir, tmpdir, worker_id, port=5433)
        return super().load_data(data_dir, tmpdir, worker_id, **kw)

    @property
    def ddl_script(self) -> Iterator[str]:
        selects = []
        for row in struct_types.abc:
            if pd.isna(row):
                datarow = "NULL"
            else:
                datarow = ", ".join(
                    "NULL" if pd.isna(val) else repr(val) for val in row.values()
                )
                datarow = f"CAST(ROW({datarow}) AS ROW(a DOUBLE, b VARCHAR, c BIGINT))"
            selects.append(f"SELECT {datarow} AS abc")

        # mirror the existing tables except for intervals which are not supported
        # and maps which we do natively in trino, because trino has more extensive
        # map support
        unsupported_memory_tables = ("intervals", "not_supported_intervals", "map")
        with self.connection.begin() as c:
            pg_tables = c.exec_driver_sql(
                f"""
                SELECT table_name
                FROM postgresql.information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name NOT IN {unsupported_memory_tables!r}
                """
            ).scalars()

        for table in pg_tables:
            dest = f"memory.default.{table}"
            yield f"DROP VIEW IF EXISTS {dest}"
            yield f"CREATE VIEW {dest} AS SELECT * FROM postgresql.public.{table}"

        yield "DROP VIEW IF EXISTS struct"
        yield f"CREATE VIEW struct AS {' UNION ALL '.join(selects)}"
        yield from super().ddl_script

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.trino.connect(
            host=TRINO_HOST,
            port=TRINO_PORT,
            user=TRINO_USER,
            password=TRINO_PASS,
            database=IBIS_TEST_TRINO_DB,
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
