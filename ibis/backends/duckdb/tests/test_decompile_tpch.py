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
from ibis.tests.util import assert_same_result, decompile_roundtrip, expose_sort_keys

tpch_catalog = {
    "lineitem": {
        "l_orderkey": "int64",
        "l_partkey": "int64",
        "l_suppkey": "int64",
        "l_linenumber": "int32",
        "l_quantity": "decimal(15, 2)",
        "l_extendedprice": "decimal(15, 2)",
        "l_discount": "decimal(15, 2)",
        "l_tax": "decimal(15, 2)",
        "l_returnflag": "string",
        "l_linestatus": "string",
        "l_shipdate": "date",
        "l_commitdate": "date",
        "l_receiptdate": "date",
        "l_shipinstruct": "string",
        "l_shipmode": "string",
        "l_comment": "string",
    },
    "customer": {
        "c_custkey": "int64",
        "c_name": "string",
        "c_address": "string",
        "c_nationkey": "int32",
        "c_phone": "string",
        "c_acctbal": "decimal(15, 2)",
        "c_mktsegment": "string",
        "c_comment": "string",
    },
    "orders": {
        "o_orderkey": "int64",
        "o_custkey": "int64",
        "o_orderstatus": "string",
        "o_totalprice": "decimal(15, 2)",
        "o_orderdate": "date",
        "o_orderpriority": "string",
        "o_clerk": "string",
        "o_shippriority": "int32",
        "o_comment": "string",
    },
    "part": {
        "p_partkey": "int64",
        "p_name": "string",
        "p_mfgr": "string",
        "p_brand": "string",
        "p_type": "string",
        "p_size": "int32",
        "p_container": "string",
        "p_retailprice": "decimal(15, 2)",
        "p_comment": "string",
    },
    "partsupp": {
        "ps_partkey": "int64",
        "ps_suppkey": "int64",
        "ps_availqty": "int32",
        "ps_supplycost": "decimal(15, 2)",
        "ps_comment": "string",
    },
    "supplier": {
        "s_suppkey": "int64",
        "s_name": "string",
        "s_address": "string",
        "s_nationkey": "int32",
        "s_phone": "string",
        "s_acctbal": "decimal(15, 2)",
        "s_comment": "string",
    },
    "nation": {
        "n_nationkey": "int32",
        "n_name": "string",
        "n_regionkey": "int32",
        "n_comment": "string",
    },
    "region": {"r_regionkey": "int32", "r_name": "string", "r_comment": "string"},
}

root = Path(__file__).absolute().parents[3]

SQL_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "h"
TPCDS_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "ds"

# Ibis has no ROLLUP / GROUPING SETS; parse_sql rejects them up front
TPCDS_ROLLUP_QUERIES = frozenset({5, 14, 18, 22, 67, 70, 77, 80, 86})

# Ibis types the mean of a decimal column and the product of two decimal columns
# at the scale of the input, where DuckDB computes a double or a wider decimal,
# and rounds the values to that scale when they are fetched (written up in
# #12190 until an issue is filed). These queries have such a column; drop a query
# from the set once it compares exactly.
TPCH_NARROW_DECIMAL_QUERIES = frozenset({1, 3, 9, 10, 15, 19})
TPCDS_NARROW_DECIMAL_QUERIES = frozenset({7, 9, 13, 26, 27, 28, 47, 53, 57, 63, 89})

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
def tpcds_catalog(con):
    return {
        name: con.table(name, database="tpcds").schema()
        for name in con.list_tables(database="tpcds")
    }


def assert_matches_duckdb(con, expr, sql, database, *, narrow_decimals=False):
    """Assert that ``expr`` computes what DuckDB computes for ``sql``."""
    ordered_sql, sort_keys = expose_sort_keys(sql, dialect="duckdb")
    with set_database(con, database):
        expected = con.con.execute(ordered_sql).df()
        result = con.to_pandas(expr)
    assert_same_result(
        result, expected, sort_keys=sort_keys, narrow_decimals=narrow_decimals
    )


@pytest.mark.parametrize(
    "tpch_query",
    [
        param(1, id="tpch01"),
        param(3, id="tpch03"),
    ],
)
def test_parse_sql_tpch(tpch_query, snapshot, con):
    tpch_query_file = SQL_QUERY_PATH / f"{tpch_query:02d}.sql"
    with open(tpch_query_file) as f:
        sql = f.read()

    expr = ibis.parse_sql(sql, tpch_catalog, dialect="duckdb")
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "out_tpch.py")

    # Import just-created snapshot
    SNAPSHOT_MODULE = f"ibis.backends.duckdb.tests.snapshots.test_decompile_tpch.test_parse_sql_tpch.tpch{tpch_query:02d}.out_tpch"
    module = importlib.import_module(SNAPSHOT_MODULE)

    assert_matches_duckdb(
        con,
        module.result,
        sql,
        "tpch",
        narrow_decimals=tpch_query in TPCH_NARROW_DECIMAL_QUERIES,
    )


@pytest.mark.parametrize("decompile", [False, True], ids=["parse", "roundtrip"])
@pytest.mark.parametrize("tpch_query", range(1, 23), ids=lambda q: f"tpch{q:02d}")
def test_parse_sql_tpch_results(tpch_query, decompile, con):
    """Every TPC-H query parses into an expression that computes the same result.

    With `decompile`, the expression is decompiled to Python and rebuilt first.
    """
    sql = (SQL_QUERY_PATH / f"{tpch_query:02d}.sql").read_text()

    expr = ibis.parse_sql(sql, tpch_catalog, dialect="duckdb")
    if decompile:
        expr = decompile_roundtrip(expr)

    assert_matches_duckdb(
        con,
        expr,
        sql,
        "tpch",
        narrow_decimals=tpch_query in TPCH_NARROW_DECIMAL_QUERIES,
    )


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
        con,
        expr,
        sql,
        "tpcds",
        narrow_decimals=tpcds_query in TPCDS_NARROW_DECIMAL_QUERIES,
    )
