from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

import ibis

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
        ("o_orderdate", "string"),
        ("o_orderpriority", "string"),
        ("o_clerk", "string"),
        ("o_shippriority", "int32"),
        ("o_comment", "string"),
    ],
}

root = Path(__file__).absolute().parents[2]

SQL_QUERY_PATH = root / "backends" / "tests" / "tpc" / "queries" / "duckdb" / "h"


@pytest.mark.parametrize(
    "tpch_query_file",
    [
        param(SQL_QUERY_PATH / "01.sql", id="tpch1"),
        param(SQL_QUERY_PATH / "03.sql", id="tpch3"),
    ],
)
def test_parse_sql_tpch(tpch_query_file, snapshot):
    with open(tpch_query_file) as f:
        sql = f.read()

    expr = ibis.parse_sql(sql, tpch_catalog)
    code = ibis.decompile(expr, format=True)
    snapshot.assert_match(code, "out.tpch.py")
