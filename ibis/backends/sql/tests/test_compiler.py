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


def test_postgres_float_literals_are_cast():
    # GH #11947: Postgres interprets bare numeric literals as `numeric`,
    # which executes as `Decimal` instead of `float`; float literals must
    # be wrapped in a cast to their ibis type
    t = ibis.table({"x": "float64"}, name="t")

    expr = t.mutate(v=ibis.literal(0.5518, type="float64"))
    sql = ibis.to_sql(expr, dialect="postgres")
    assert "CAST(0.5518 AS DOUBLE PRECISION)" in sql

    # non-finite values are still rendered as casted string literals
    nan_sql = ibis.to_sql(
        t.mutate(v=ibis.literal(float("nan"), type="float64")), dialect="postgres"
    )
    assert "CAST('NaN' AS DOUBLE PRECISION)" in nan_sql
