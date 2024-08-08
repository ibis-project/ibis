from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path

import pytest
from pytest import param

import ibis
from ibis.backends.tests.tpc.conftest import compare_tpc_results
from ibis.formats.pandas import PandasData

tpch_catalog = {
    "lineitem": {
        "l_orderkey": "int32",
        "l_partkey": "int32",
        "l_suppkey": "int32",
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
    "customer": [
        ("c_custkey", "int64"),
        ("c_name", "string"),
        ("c_address", "string"),
        ("c_nationkey", "int16"),
        ("c_phone", "string"),
        ("c_acctbal", "decimal"),
        ("c_mktsegment", "string"),
        ("c_comment", "string"),
    ],
    "orders": [
        ("o_orderkey", "int64"),
        ("o_custkey", "int64"),
        ("o_orderstatus", "string"),
        ("o_totalprice", "decimal(12,2)"),
        ("o_orderdate", "date"),
        ("o_orderpriority", "string"),
        ("o_clerk", "string"),
        ("o_shippriority", "int32"),
        ("o_comment", "string"),
    ],
}

root = Path(__file__).absolute().parents[3]

SQL_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "h"


@contextmanager
def set_database(con, db):
    olddb = con.current_database
    con.raw_sql(f"USE {db}")
    yield
    con.raw_sql(f"USE {olddb}")


@pytest.mark.parametrize(
    "tpch_query",
    [
        param(1, id="tpch01"),
        param(3, id="tpch03"),
    ],
)
def test_parse_sql_tpch(tpch_query, snapshot, con, data_dir):
    tpch_query_file = SQL_QUERY_PATH / f"{tpch_query:02d}.sql"
    with open(tpch_query_file) as f:
        sql = f.read()

    expr = ibis.parse_sql(sql, tpch_catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "out_tpch.py")

    # Import just-created snapshot
    SNAPSHOT_MODULE = f"ibis.backends.duckdb.tests.snapshots.test_decompile_tpch.test_parse_sql_tpch.tpch{tpch_query:02d}.out_tpch"
    module = importlib.import_module(SNAPSHOT_MODULE)

    with set_database(con, "tpch"):
        # Get results from executing SQL directly on DuckDB
        expected_df = con.con.execute(sql).df()
        # Get results from decompiled ibis query
        result_df = con.to_pandas(module.result)

    # Then set the expected columns so we can coerce the datatypes
    # of the pandas dataframe correctly
    expected_df.columns = result_df.columns

    expected_df = PandasData.convert_table(expected_df, module.result.schema())

    compare_tpc_results(result_df, expected_df)
