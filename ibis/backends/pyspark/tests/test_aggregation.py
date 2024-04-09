from __future__ import annotations

import pytest

import ibis
from ibis.backends.tests.errors import PySparkAnalysisException

pyspark = pytest.importorskip("pyspark")


@pytest.mark.parametrize(
    "table_name",
    [
        pytest.param("null_table", id="batch"),
        pytest.param(
            "null_table_streaming",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Streaming aggregations require watermark.",
            ),
            id="streaming",
        ),
    ],
)
@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        (lambda t: t.age.count(), lambda t: len(t.age.dropna())),
        (lambda t: t.age.sum(), lambda t: t.age.sum()),
    ],
    ids=["count", "sum"],
)
def test_aggregation_float_nulls(con, table_name, result_fn, expected_fn, monkeypatch):
    monkeypatch.setattr(ibis.options.pyspark, "treat_nan_as_null", True)

    table = con.table(table_name)
    df = table.execute()

    expr = result_fn(table)
    result = expr.execute()

    expected = expected_fn(df)
    assert pytest.approx(expected) == result
