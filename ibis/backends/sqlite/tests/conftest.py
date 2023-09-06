from __future__ import annotations

import contextlib
import csv
import io
import sqlite3
from typing import Any

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_structs = False
    stateful = False
    deps = ("sqlalchemy",)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.sqlite.connect(**kw)

    def _load_data(self, **kw: Any) -> None:
        """Load test data into a SQLite backend instance."""
        super()._load_data(**kw)

        with self.connection.begin() as con:
            for table in TEST_TABLES:
                basename = f"{table}.csv"
                with self.data_dir.joinpath("csv", basename).open(
                    "r", encoding="UTF-8"
                ) as f:
                    if basename == "astronauts.csv":
                        input = io.StringIO(f.read().replace("\n ", " "))
                    else:
                        input = f
                    reader = csv.reader(input)
                    header = next(reader)
                    assert header, f"empty header for table: `{table}`"
                    spec = ", ".join("?" * len(header))
                    with contextlib.closing(con.connection.cursor()) as cur:
                        cur.executemany(f"INSERT INTO {table} VALUES ({spec})", reader)

    @property
    def functional_alltypes(self) -> ir.Table:
        t = super().functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast("timestamp"))


@pytest.fixture
def dbpath(tmp_path):
    path = tmp_path / "test.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t AS SELECT 1 a UNION SELECT 2 UNION SELECT 3")
    con.execute("CREATE TABLE s AS SELECT 1 b UNION SELECT 2")
    return path


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


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
        .compile(dialect=dialect, compile_kwargs={"literal_binds": True})
    )


@pytest.fixture(scope="session")
def sqla_compile(dialect):
    return lambda expr: str(
        expr.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
    )


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_sqla(con, alltypes):
    name = alltypes.op().name
    return con._get_sqla_table(name)


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()
