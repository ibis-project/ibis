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


def test_mssql_stdev_roundtrips_as_sample():
    # GH #12057: sqlglot parses T-SQL STDEV (sample) into sge.Stddev. The MSSQL
    # dialect used to render that node as STDEVP (population), so a same-
    # dialect round trip through Table.sql() / add_query_to_expr silently
    # changed the statistic. Bare Stddev must render as STDEV; StddevPop
    # already covers STDEVP.
    (result,) = sg.transpile(
        "SELECT STDEV([x]) AS [s] FROM [t]", read=MSSQL, write=MSSQL
    )
    assert result == "SELECT STDEV([x]) AS [s] FROM [t]"


def test_mssql_std_how_emits_distinct_functions():
    t = ibis.table({"x": "float"}, name="t")
    sample_sql = ibis.to_sql(t.x.std(how="sample"), dialect="mssql")
    pop_sql = ibis.to_sql(t.x.std(how="pop"), dialect="mssql")
    assert "STDEV(" in sample_sql
    assert "STDEVP(" not in sample_sql
    assert "STDEVP(" in pop_sql
