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


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, 'ELEMENT_AT("t0"."s", 1)'),
        (1, 'ELEMENT_AT("t0"."s", 2)'),
        (-1, 'ELEMENT_AT("t0"."s", -1)'),
        (-2, 'ELEMENT_AT("t0"."s", -2)'),
    ],
)
def test_trino_array_index_literal(index, expected):
    t = ibis.table({"s": "array<string>"}, name="t")
    expr = t.select(res=t.s[index])
    sql = ibis.to_sql(expr, dialect="trino")
    assert expected in sql


def test_trino_array_index_dynamic():
    t = ibis.table({"s": "array<string>", "idx": "int64"}, name="t")
    expr = t.select(res=t.s[t.idx])
    sql = ibis.to_sql(expr, dialect="trino")
    assert (
        'ELEMENT_AT("t0"."s", IF("t0"."idx" >= 0, "t0"."idx" + 1, "t0"."idx"))' in sql
    )


def test_athena_array_index():
    t = ibis.table({"s": "array<string>"}, name="t")
    expr = t.select(res=t.s[-1])
    sql = ibis.to_sql(expr, dialect="athena")
    assert 'ELEMENT_AT("t0"."s", -1)' in sql
