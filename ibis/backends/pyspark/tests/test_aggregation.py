from __future__ import annotations

import pytest

import ibis

pytest.importorskip("pyspark")


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        (lambda t: t.age.count(), lambda t: len(t.age.dropna())),
        (lambda t: t.age.sum(), lambda t: t.age.sum()),
    ],
    ids=["count", "sum"],
)
def test_aggregation_float_nulls(con, result_fn, expected_fn, monkeypatch):
    monkeypatch.setattr(ibis.options.pyspark, "treat_nan_as_null", True)

    table = con.table("null_table")
    df = table.compile().toPandas()

    expr = result_fn(table)
    result = expr.execute()

    expected = expected_fn(df)
    assert pytest.approx(expected) == result
