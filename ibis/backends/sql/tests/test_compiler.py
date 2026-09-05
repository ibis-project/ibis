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


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, 'ELEMENT_AT("t0"."arr", 1)'),
        (1, 'ELEMENT_AT("t0"."arr", 2)'),
        (2, 'ELEMENT_AT("t0"."arr", 3)'),
        (-1, 'ELEMENT_AT("t0"."arr", -1)'),
        (-2, 'ELEMENT_AT("t0"."arr", -2)'),
        (-3, 'ELEMENT_AT("t0"."arr", -3)'),
    ],
)
def test_trino_array_index_literal(index, expected):
    t = ibis.table({"arr": "array<string>"}, name="t")
    sql = ibis.to_sql(t.select(res=t.arr[index]), dialect="trino")
    assert expected in sql


def test_trino_array_index_dynamic():
    t = ibis.table({"arr": "array<string>", "idx": "int64"}, name="t")
    sql = ibis.to_sql(t.select(res=t.arr[t.idx]), dialect="trino")
    assert (
        'ELEMENT_AT("t0"."arr", IF("t0"."idx" >= 0, "t0"."idx" + 1, "t0"."idx"))' in sql
    )


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, 'ELEMENT_AT("t0"."arr", 1)'),
        (1, 'ELEMENT_AT("t0"."arr", 2)'),
        (-1, 'ELEMENT_AT("t0"."arr", -1)'),
        (-2, 'ELEMENT_AT("t0"."arr", -2)'),
    ],
)
def test_athena_array_index(index, expected):
    t = ibis.table({"arr": "array<string>"}, name="t")
    sql = ibis.to_sql(t.select(res=t.arr[index]), dialect="athena")
    assert expected in sql
