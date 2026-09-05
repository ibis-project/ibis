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
@pytest.mark.parametrize(
    ("col", "expected_pattern"),
    [
        ("s", 'JSON_FORMAT(CAST("t0"."s" AS JSON))'),
        ("m", 'JSON_FORMAT(CAST("t0"."m" AS JSON))'),
        ("arr", 'JSON_FORMAT(CAST("t0"."arr" AS JSON))'),
    ],
)
def test_trino_nested_cast_to_string(dialect, col, expected_pattern):
    t = ibis.table(
        {
            "s": "struct<a: int, b: string>",
            "m": "map<string, int>",
            "arr": "array<int>",
        },
        name="t",
    )
    expr = t.select(res=t[col].cast("string"))
    sql = ibis.to_sql(expr, dialect=dialect)
    assert expected_pattern in sql


def test_trino_json_cast_to_string():
    t = ibis.table({"j": "json"}, name="t")
    expr = t.select(res=t.j.cast("string"))
    sql = ibis.to_sql(expr, dialect="trino")
    assert 'JSON_FORMAT("t0"."j")' in sql


@pytest.mark.parametrize("dialect", ["trino", "athena"])
def test_trino_nested_try_cast_to_string(dialect):
    t = ibis.table({"s": "struct<a: int>"}, name="t")
    expr = t.select(res=t.s.try_cast("string"))
    sql = ibis.to_sql(expr, dialect=dialect)
    assert 'TRY(JSON_FORMAT(CAST("t0"."s" AS JSON)))' in sql
