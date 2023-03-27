from __future__ import annotations

import contextlib
import csv
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend


class TestConf(BackendTest, RoundAwayFromZero):
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    check_dtype = False
    returned_timestamp_unit = 's'
    supports_structs = False

    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

        schema = data_directory.parent.joinpath('schema', 'sqlite.sql').read_text()

        with self.connection.begin() as con:
            for stmt in filter(None, map(str.strip, schema.split(';'))):
                con.exec_driver_sql(stmt)

            for table in TEST_TABLES:
                basename = f"{table}.csv"
                with data_directory.joinpath(basename).open("r") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    assert header, f"empty header for table: `{table}`"
                    spec = ", ".join("?" * len(header))
                    with contextlib.closing(con.connection.cursor()) as cur:
                        cur.executemany(f"INSERT INTO {table} VALUES ({spec})", reader)

    @staticmethod
    def _load_data(
        data_dir: Path, script_dir: Path, database: str | None = None, **_: Any
    ) -> None:
        """Load test data into a SQLite backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        return TestConf(data_dir)

    @staticmethod
    def connect(data_directory: Path) -> BaseBackend:
        return ibis.sqlite.connect()  # type: ignore

    @property
    def functional_alltypes(self) -> ir.Table:
        t = super().functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast('timestamp'))


@pytest.fixture
def dbpath(tmp_path):
    path = tmp_path / "test.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t AS SELECT 1 a UNION SELECT 2 UNION SELECT 3")
    con.execute("CREATE TABLE s AS SELECT 1 b UNION SELECT 2")
    return path


@pytest.fixture(scope="session")
def con(data_directory):
    return TestConf(data_directory).connection


@pytest.fixture(scope="session")
def dialect():
    import sqlalchemy as sa

    return sa.dialects.sqlite.dialect()


@pytest.fixture(scope="session")
def translate(dialect):
    from ibis.backends.sqlite import Backend

    context = Backend.compiler.make_context()
    return lambda expr: str(
        Backend.compiler.translator_class(expr, context)
        .get_result()
        .compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture(scope="session")
def sqla_compile(dialect):
    return lambda expr: str(
        expr.compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_sqla(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()
