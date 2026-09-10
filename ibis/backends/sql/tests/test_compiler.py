from __future__ import annotations

import pytest
import sqlglot as sg

import ibis
from ibis import _
from ibis.backends.sql.compilers._compat import Drop
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
    # pin the dialect: letting `to_sql` pick one sets the global
    # `ibis.options.default_backend`, which leaks into later tests
    assert ibis.to_sql(expr, dialect="duckdb")


@pytest.mark.core
def test_transpile_join():
    (result,) = sg.transpile(
        "SELECT * FROM t1 JOIN t2 ON x = y", read="duckdb", write=Trino
    )
    assert "CROSS JOIN" not in result


@pytest.mark.core
def test_drop_includes_the_table():
    # GH #12101: sqlglot 30.18 renamed Drop's this kwarg to tables
    stmt = Drop(kind="TABLE", this=sg.table("t", quoted=True), exists=True)
    assert stmt.sql("duckdb") == 'DROP TABLE IF EXISTS "t"'
