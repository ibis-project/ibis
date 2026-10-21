from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path

import pytest
from pytest import param

import ibis
from ibis.tests.util import (
    assert_same_result,
    decompile_roundtrip,
    expose_sort_keys,
    unnamed_columns,
)

root = Path(__file__).absolute().parents[3]

SQL_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "h"
TPCDS_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "ds"

# the queries that return no rows on the test data, as in the TPC-DS suite
TPCDS_EMPTY_QUERIES = frozenset({24, 25, 54, 64, 65, 73, 77, 82, 91, 93})

# the queries the planner-based ``parse_sql`` translates today; every other
# query is a strict xfail until the converter that replaces the planner lands
TPCH_TRANSLATED_QUERIES = frozenset({1, 3, 5, 6, 10, 19})
TPCDS_TRANSLATED_QUERIES = frozenset({7, 13, 26, 28, 37, 42, 48, 82, 96})

# translated queries whose result still differs, and why
TPCDS_MISTRANSLATED_QUERIES = {
    28: "parse_sql turns COUNT(x) and COUNT(DISTINCT x) into COUNT(*)",
    37: "parse_sql drops a GROUP BY that has no aggregates",
}


def _not_yet(translated: bool, mistranslated: str | None = None) -> list:
    """The marks of a query ``parse_sql`` does (not) translate yet.

    ``mistranslated`` is why a translated query's result still differs.
    """
    if mistranslated is not None:
        return [pytest.mark.xfail(reason=mistranslated, strict=True)]
    if translated:
        return []
    return [
        pytest.mark.xfail(
            reason="parse_sql cannot translate this query yet", strict=True
        )
    ]


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
@pytest.mark.parametrize(
    "tpch_query",
    [
        param(q, id=f"tpch{q:02d}", marks=_not_yet(q in TPCH_TRANSLATED_QUERIES))
        for q in range(1, 23)
    ],
)
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
            marks=_not_yet(
                q in TPCDS_TRANSLATED_QUERIES, TPCDS_MISTRANSLATED_QUERIES.get(q)
            ),
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
