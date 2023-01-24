from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES, _random_identifier
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.backends.tests.data import struct_types
from ibis.util import consume

TRINO_USER = os.environ.get(
    'IBIS_TEST_TRINO_USER', os.environ.get('TRINO_USER', 'user')
)
TRINO_PASS = os.environ.get(
    'IBIS_TEST_TRINO_PASSWORD', os.environ.get('TRINO_PASSWORD', '')
)
TRINO_HOST = os.environ.get(
    'IBIS_TEST_TRINO_HOST', os.environ.get('TRINO_HOST', 'localhost')
)
TRINO_PORT = os.environ.get('IBIS_TEST_TRINO_PORT', os.environ.get('TRINO_PORT', 8080))
IBIS_TEST_TRINO_DB = os.environ.get(
    'IBIS_TEST_TRINO_DATABASE',
    os.environ.get('TRINO_DATABASE', 'memory'),
)

sa = pytest.importorskip("sqlalchemy")


class TestConf(BackendTest, RoundAwayFromZero):
    # trino rounds half to even for double precision and half away from zero
    # for numeric and decimal

    returned_timestamp_unit = 's'
    supports_structs = True

    @staticmethod
    def _load_data(data_dir: Path, script_dir: Path, **_: Any) -> None:
        """Load test data into a Trino backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        from ibis.backends.postgres.tests.conftest import (
            IBIS_TEST_POSTGRES_DB,
            PG_HOST,
            PG_PASS,
            PG_USER,
        )
        from ibis.backends.postgres.tests.conftest import TestConf as PostgresTestConf

        PostgresTestConf._load_data(data_dir, script_dir, port=5433)
        pgcon = ibis.postgres.connect(
            host=PG_HOST,
            port=5433,
            user=PG_USER,
            password=PG_PASS,
            database=IBIS_TEST_POSTGRES_DB,
        )

        con = TestConf.connect(data_dir)

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
        unsupported_memory_tables = {"intervals", "not_supported_intervals", "map"}
        lines = []
        for table in frozenset(pgcon.list_tables()) - unsupported_memory_tables:
            dest = f"memory.default.{table}"
            lines.append(f"DROP VIEW IF EXISTS {dest}")
            lines.append(
                f"CREATE VIEW {dest} AS SELECT * FROM postgresql.public.{table}"
            )

        lines.extend(
            itertools.chain(
                [
                    "DROP VIEW IF EXISTS struct",
                    f"CREATE VIEW struct AS {' UNION ALL '.join(selects)}",
                ],
                Path(script_dir, "schema", "trino.sql").read_text().split(";"),
            )
        )

        with con.begin() as c:
            consume(map(c.exec_driver_sql, filter(None, map(str.strip, lines))))

    @staticmethod
    def connect(data_directory: Path):
        return ibis.trino.connect(
            host=TRINO_HOST,
            port=TRINO_PORT,
            user=TRINO_USER,
            password=TRINO_PASS,
            database=IBIS_TEST_TRINO_DB,
            schema="default",
        )

    def _remap_column_names(self, table_name: str) -> dict[str, str]:
        table = self.connection.table(table_name)
        return table.relabel(
            dict(zip(table.schema().names, TEST_TABLES[table_name].names))
        )

    @property
    def batting(self):
        return self._remap_column_names("batting")

    @property
    def awards_players(self):
        return self._remap_column_names("awards_players")


@pytest.fixture(scope='session')
def con(tmp_path_factory, data_directory, script_directory, worker_id):
    return TestConf.load_data(
        data_directory,
        script_directory,
        tmp_path_factory,
        worker_id,
    ).connect(data_directory)


@pytest.fixture(scope='module')
def db(con):
    return con.database()


@pytest.fixture(scope='module')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='module')
def geotable(con):
    return con.table('geo')


@pytest.fixture(scope='module')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='module')
def gdf(geotable):
    return geotable.execute()


@pytest.fixture(scope='module')
def at(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture(scope='module')
def intervals(con):
    return con.table("intervals")


@pytest.fixture
def translate():
    from ibis.backends.trino import Backend

    context = Backend.compiler.make_context()
    return lambda expr: (Backend.compiler.translator_class(expr, context).get_result())


@pytest.fixture
def temp_table(con) -> Generator[str, None, None]:
    """Return a temporary table name."""
    name = _random_identifier('table')
    try:
        yield name
    finally:
        con.drop_table(name, force=True)
