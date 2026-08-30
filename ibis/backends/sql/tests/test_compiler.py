from __future__ import annotations

from datetime import datetime, timezone

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


def test_trino_timezone_cast_uses_timezone_functions():
    string_expr = ibis.memtable({"x": ["2023-01-02"]}).select(
        casted=_.x.cast("timestamp('Europe/Paris')")
    )
    timestamp_expr = ibis.memtable(
        {"x": [datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc)]},
        schema=ibis.schema({"x": "timestamp('UTC')"}),
    ).select(casted=_.x.cast("timestamp('Europe/Paris')"))

    string_sql = ibis.to_sql(string_expr, dialect="trino")
    timestamp_sql = ibis.to_sql(timestamp_expr, dialect="trino")

    assert "WITH_TIMEZONE(" in string_sql
    assert "AT_TIMEZONE(" not in string_sql
    assert "AT_TIMEZONE(" in timestamp_sql
