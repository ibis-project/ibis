"""Tests for JSON operations."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pandas.testing as tm
import pytest
from packaging.version import parse as vparse
from pytest import param

pytestmark = [
    pytest.mark.never(["impala"], reason="doesn't support JSON and never will"),
    pytest.mark.notyet(["clickhouse"], reason="upstream is broken"),
    pytest.mark.notimpl(["datafusion", "mssql", "druid", "oracle"]),
]


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        param(
            lambda t: t.js["a"].name("res"),
            pd.Series([[1, 2, 3, 4], None, "foo"] + [None] * 3, name="res"),
            id="object",
        ),
        param(
            lambda t: t.js[1].name("res"),
            pd.Series([None] * 4 + [47, None], dtype="object", name="res"),
            id="array",
        ),
    ],
)
@pytest.mark.notyet(
    ["sqlite"],
    condition=vparse(sqlite3.sqlite_version) < vparse("3.38.0"),
    reason="JSON not supported in SQLite < 3.38.0",
)
def test_json_getitem(json_t, expr_fn, expected):
    expr = expr_fn(json_t)
    result = expr.execute()
    tm.assert_series_equal(result.fillna(pd.NA), expected.fillna(pd.NA))


@pytest.mark.notimpl(["dask", "mysql", "pandas"])
@pytest.mark.notyet(["bigquery", "sqlite"], reason="doesn't support maps")
@pytest.mark.notyet(["postgres"], reason="only supports map<string, string>")
@pytest.mark.notyet(
    ["pyspark", "trino"], reason="should work but doesn't deserialize JSON"
)
def test_json_map(json_t):
    expr = json_t.js.map.name("res")
    result = expr.execute()
    expected = pd.Series(
        [
            {"a": [1, 2, 3, 4], "b": 1},
            {"a": None, "b": 2},
            {"a": "foo", "c": None},
            None,
            None,
            None,
        ],
        dtype="object",
        name="res",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask", "mysql", "pandas"])
@pytest.mark.notyet(["sqlite"], reason="doesn't support arrays")
@pytest.mark.notyet(
    ["pyspark", "trino"], reason="should work but doesn't deserialize JSON"
)
@pytest.mark.notyet(["bigquery"], reason="doesn't allow null in arrays")
def test_json_array(json_t):
    expr = json_t.js.array.name("res")
    result = expr.execute()
    expected = pd.Series(
        [None, None, None, None, [42, 47, 55], []], name="res", dtype="object"
    )
    tm.assert_series_equal(result, expected)
