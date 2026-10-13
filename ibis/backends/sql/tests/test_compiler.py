from __future__ import annotations

import pytest
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


def test_visit_window_function_override_is_used_for_windowed_function():
    from ibis.backends.sql.compilers.duckdb import DuckDBCompiler

    with pytest.warns(FutureWarning, match="visit_WindowedFunction"):

        class MyCompiler(DuckDBCompiler):
            def visit_WindowFunction(self, op, **kwargs):  # noqa: ARG002
                return sg.exp.convert("my_window")

    t = ibis.table({"a": "int64"}, name="t")
    expr = t.select(s=t.a.sum().over())
    sql = MyCompiler().to_sqlglot(expr).sql()
    assert "my_window" in sql
