from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlglot
from packaging.version import parse as vparse
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis.tests.util import (
    assert_same_result,
    decompile_roundtrip,
    expose_sort_keys,
    unnamed_columns,
)

root = Path(__file__).absolute().parents[3]

SQL_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "h"
TPCDS_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "ds"

# Ibis has no ROLLUP / GROUPING SETS; parse_sql rejects them up front
TPCDS_ROLLUP_QUERIES = frozenset({5, 14, 18, 22, 67, 70, 77, 80, 86})

# the queries that return no rows on the test data, as in the TPC-DS suite
TPCDS_EMPTY_QUERIES = frozenset({24, 25, 54, 64, 65, 73, 77, 82, 91, 93})

# sqlglot has no ``Rank`` class before 27.11, so parse_sql rejects RANK() there;
# these queries use it (67, 70 and 86 do too, behind the ROLLUP mark)
OLD_SQLGLOT = vparse(sqlglot.__version__) < vparse("27.11")
TPCDS_RANKING_QUERIES = frozenset({36, 44, 47, 49, 57})


@contextmanager
def set_database(con, db):
    olddb = con.current_database
    con.raw_sql(f"USE {db}")
    yield
    con.raw_sql(f"USE {olddb}")


@pytest.fixture(scope="session")
def tpch_catalog(con):
    return {
        name: con.table(name, database="tpch").schema()
        for name in con.list_tables(database="tpch")
    }


@pytest.fixture(scope="session")
def tpcds_catalog(con):
    return {
        name: con.table(name, database="tpcds").schema()
        for name in con.list_tables(database="tpcds")
    }


def assert_matches_duckdb(con, expr, sql, database, *, empty=False):
    """Assert that the SQL Ibis compiles ``expr`` to computes what ``sql`` does.

    The query must return rows, so that the comparison shows something, unless
    it is known to be ``empty``.
    """
    ordered_sql, sort_keys = expose_sort_keys(sql, dialect="duckdb")
    with set_database(con, database):
        expected = con.con.execute(ordered_sql).fetch_arrow_table()
        result = con.con.execute(con.compile(expr)).fetch_arrow_table()
    assert (expected.num_rows == 0) == empty, (
        f"the query returns {expected.num_rows} rows"
    )
    assert_same_result(
        result,
        expected,
        sort_keys=sort_keys,
        unnamed=unnamed_columns(sql, dialect="duckdb"),
    )


@pytest.mark.parametrize(
    "tpch_query",
    [
        param(1, id="tpch01"),
        param(3, id="tpch03"),
    ],
)
def test_parse_sql_tpch(tpch_query, snapshot, con, tpch_catalog):
    tpch_query_file = SQL_QUERY_PATH / f"{tpch_query:02d}.sql"
    with open(tpch_query_file) as f:
        sql = f.read()

    expr = ibis.parse_sql(sql, tpch_catalog, dialect="duckdb")
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "out_tpch.py")

    # Import just-created snapshot
    SNAPSHOT_MODULE = f"ibis.backends.duckdb.tests.snapshots.test_decompile_tpch.test_parse_sql_tpch.tpch{tpch_query:02d}.out_tpch"
    module = importlib.import_module(SNAPSHOT_MODULE)

    assert_matches_duckdb(con, module.result, sql, "tpch")


@pytest.mark.parametrize("decompile", [False, True], ids=["parse", "roundtrip"])
@pytest.mark.parametrize("tpch_query", range(1, 23), ids=lambda q: f"tpch{q:02d}")
def test_parse_sql_tpch_results(tpch_query, decompile, con, tpch_catalog):
    """Every TPC-H query parses into an expression that computes the same result.

    With `decompile`, the expression is decompiled to Python and rebuilt first.
    """
    sql = (SQL_QUERY_PATH / f"{tpch_query:02d}.sql").read_text()

    expr = ibis.parse_sql(sql, tpch_catalog, dialect="duckdb")
    if decompile:
        expr = decompile_roundtrip(expr)

    assert_matches_duckdb(con, expr, sql, "tpch")


@pytest.mark.parametrize("decompile", [False, True], ids=["parse", "roundtrip"])
@pytest.mark.parametrize(
    "tpcds_query",
    [
        param(
            q,
            id=f"tpcds{q:02d}",
            marks=[
                pytest.mark.xfail(
                    raises=com.UnsupportedOperationError,
                    reason="GROUP BY ROLLUP is not supported",
                    strict=True,
                )
            ]
            if q in TPCDS_ROLLUP_QUERIES
            else [
                pytest.mark.skipif(
                    OLD_SQLGLOT and q in TPCDS_RANKING_QUERIES,
                    reason="sqlglot has no ranking function classes before 27.11",
                )
            ],
        )
        for q in range(1, 100)
    ],
)
def test_parse_sql_tpcds_results(tpcds_query, decompile, con, tpcds_catalog):
    """Every TPC-DS query parses into an expression that computes the same result.

    With `decompile`, the expression is decompiled to Python and rebuilt first.
    """
    sql = (TPCDS_QUERY_PATH / f"{tpcds_query:02d}.sql").read_text()

    expr = ibis.parse_sql(sql, tpcds_catalog, dialect="duckdb")
    if decompile:
        expr = decompile_roundtrip(expr)

    assert_matches_duckdb(
        con, expr, sql, "tpcds", empty=tpcds_query in TPCDS_EMPTY_QUERIES
    )
