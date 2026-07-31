from __future__ import annotations

import sqlglot as sg

import ibis
from ibis import _
from ibis.backends.sql.dialects import MSSQL, Trino


def test_window_with_row_number_compiles():
    # GH #8058: the add_order_by_to_empty_ranking_window_functions rule was
    # matching on `RankBase` subclasses with a pattern expecting an `arg`
    # attribute, which is not present on `RowNumber`
    expr = (
        ibis.memtable({"a": range(30)})
        .mutate(id=ibis.row_number())
        .sample(fraction=0.25, seed=0)
        .mutate(is_test=_.id.isin(_.id))
        .filter(~_.is_test)
    )
    assert ibis.to_sql(expr)


def test_transpile_join():
    (result,) = sg.transpile(
        "SELECT * FROM t1 JOIN t2 ON x = y", read="duckdb", write=Trino
    )
    assert "CROSS JOIN" not in result


def test_mssql_stdev_roundtrips_to_sample():
    # GH #12057: the bare Stddev node is T-SQL's sample STDEV, so rendering it
    # as STDEVP silently changed the statistic on a same-dialect round trip
    (result,) = sg.transpile(
        "SELECT STDEV([x]) AS [s] FROM [t]", read=MSSQL, write=MSSQL
    )
    assert result == "SELECT STDEV([x]) AS [s] FROM [t]"


def test_mssql_std_emits_distinct_functions():
    # the two ibis spellings must not collapse onto the same T-SQL function
    t = ibis.table({"x": "float"}, name="t")
    assert "STDEV(" in ibis.to_sql(t.x.std(how="sample"), dialect="mssql")
    assert "STDEVP(" in ibis.to_sql(t.x.std(how="pop"), dialect="mssql")
