"""Tests for JSON operations."""

import pandas as pd
import pytest
from pytest import param


@pytest.mark.notimpl(["bigquery", "datafusion", "mssql"])
@pytest.mark.notyet(["clickhouse"], reason="upstream is broken")
@pytest.mark.never(["impala"], reason="doesn't support JSON and never will")
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
            id="getitem_object",
            marks=[pytest.mark.min_server_version(sqlite="3.38.0")],
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
def test_json_getitem(backend, json_t, expr_fn, expected):
    expr = expr_fn(json_t)
    result = expr.execute()
    backend.assert_series_equal(result, expected)
