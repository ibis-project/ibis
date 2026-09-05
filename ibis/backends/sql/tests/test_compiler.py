from __future__ import annotations

import sqlglot as sg

import ibis
from ibis import _
from ibis.backends.sql.dialects import Trino


def test_window_with_row_number_compiles():
    # GH #8058: the add_order_by_to_empty_ranking_window_functions rule was
    # matching on `RankBase` subclasses with a pattern expecting an `arg`
    # attribute, which is not present on `RowNumber`
    expr = (
        ibis.memtable({"a": range(30)})
        .mutate(id=ibis.row_number())
        .sample(0.25, seed=0)
        .mutate(is_test=_.id.isin(_.id))
        .filter(~_.is_test)
    )
    assert ibis.to_sql(expr)


def test_transpile_join():
    (result,) = sg.transpile(
        "SELECT * FROM t1 JOIN t2 ON x = y", read="duckdb", write=Trino
    )
    assert "CROSS JOIN" not in result


def test_mssql_stddev_transpile():
    from ibis.backends.sql.dialects import MSSQL

    (stdev_result,) = sg.transpile(
        "SELECT STDEV([x]) AS [s] FROM [t]", read=MSSQL, write=MSSQL
    )
    assert "STDEV([x])" in stdev_result
    assert "STDEVP" not in stdev_result

    (stdevp_result,) = sg.transpile(
        "SELECT STDEVP([x]) AS [s] FROM [t]", read=MSSQL, write=MSSQL
    )
    assert "STDEVP([x])" in stdevp_result
