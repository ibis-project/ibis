from __future__ import annotations

import pytest
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


@pytest.mark.parametrize("func", ["STDEV", "STDEVP", "VAR", "VARP"])
def test_mssql_stat_funcs_roundtrip(func):
    # GH #12057: bare Stddev was mapped to STDEVP, turning the sample deviation
    # into the population one when raw T-SQL is round-tripped
    sql = f"SELECT {func}([x]) AS [s] FROM [t]"
    assert sg.transpile(sql, read=MSSQL, write=MSSQL)[0] == sql
