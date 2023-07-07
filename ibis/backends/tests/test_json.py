"""Tests for JSON operations."""
from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

from ibis.common.exceptions import OperationNotDefinedError

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
            pd.Series(
                [[1, 2, 3, 4], None, "foo", None, None, None],
                name="res",
                dtype="object",
            ),
            marks=[pytest.mark.min_server_version(sqlite="3.38.0")],
            id="getitem_object",
        ),
        param(
            lambda t: t.js[1].name("res"),
            pd.Series(
                [None, None, None, None, 47, None],
                dtype="object",
                name="res",
            ),
            marks=[pytest.mark.min_server_version(sqlite="3.38.0")],
            id="getitem_array",
        ),
    ],
)
def test_json_getitem(json_t, expr_fn, expected):
    expr = expr_fn(json_t)
    result = expr.execute()
    tm.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask", "mysql", "pandas"])
@pytest.mark.notyet(["bigquery", "sqlite"], reason="doesn't support maps")
@pytest.mark.notyet(["postgres"], reason="only supports map<string, string>")
@pytest.mark.notyet(
    ["pyspark", "trino"], reason="should work but doesn't deserialize JSON"
)
@pytest.mark.notimpl(["duckdb"], raises=OperationNotDefinedError)
def test_json_map(json_t):
    expr = json_t.js.map.name("res")
    result = expr.execute()
    expected = pd.Series(
        [
            {'a': [1, 2, 3, 4], 'b': 1},
            {'a': None, 'b': 2},
            {'a': 'foo', 'c': None},
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
@pytest.mark.notimpl(["duckdb"], raises=OperationNotDefinedError)
def test_json_array(json_t):
    expr = json_t.js.array.name("res")
    result = expr.execute()
    expected = pd.Series(
        [None, None, None, None, [42, 47, 55], []], name="res", dtype="object"
    )
    tm.assert_series_equal(result, expected)
