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


@pytest.mark.parametrize("dialect", ["trino", "athena"])
def test_trino_string_cast_to_timestamp(dialect):
    t = ibis.table({"s": "string"}, name="t")
    expr = t.select(res=t.s.cast("timestamp"))
    sql = ibis.to_sql(expr, dialect=dialect)
    assert 'CAST(FROM_ISO8601_TIMESTAMP("t0"."s") AS TIMESTAMP)' in sql

    expr_tz = t.select(res=t.s.cast("timestamp('UTC')"))
    sql_tz = ibis.to_sql(expr_tz, dialect=dialect)
    assert (
        'CAST(FROM_ISO8601_TIMESTAMP("t0"."s") AS TIMESTAMP WITH TIME ZONE)' in sql_tz
    )

    expr_try = t.select(res=t.s.try_cast("timestamp"))
    sql_try = ibis.to_sql(expr_try, dialect=dialect)
    assert 'TRY(CAST(FROM_ISO8601_TIMESTAMP("t0"."s") AS TIMESTAMP))' in sql_try
