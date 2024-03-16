"""Tests for JSON operations."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as vparse

pytestmark = [
    pytest.mark.never(["impala"], reason="doesn't support JSON and never will"),
    pytest.mark.notyet(["clickhouse"], reason="upstream is broken"),
    pytest.mark.notimpl(["datafusion", "exasol", "mssql", "druid", "oracle"]),
]


@pytest.mark.notyet(
    ["sqlite"],
    condition=vparse(sqlite3.sqlite_version) < vparse("3.38.0"),
    reason="JSON not supported in SQLite < 3.38.0",
)
@pytest.mark.broken(
    ["flink"],
    reason="https://github.com/ibis-project/ibis/pull/6920#discussion_r1373212503",
)
@pytest.mark.broken(
    ["risingwave"], reason="TODO(Kexiang): order mismatch in array", strict=False
)
def test_json_getitem_object(json_t):
    expr_fn = lambda t: t.js["a"].name("res")
    expected = frozenset([(1, 2, 3, 4), None, "foo"] + [None] * 3)
    expr = expr_fn(json_t)
    result = frozenset(
        expr.execute()
        .map(lambda o: tuple(o) if isinstance(o, list) else o)
        .replace({np.nan: None})
    )
    assert result == expected


@pytest.mark.notyet(
    ["sqlite"],
    condition=vparse(sqlite3.sqlite_version) < vparse("3.38.0"),
    reason="JSON not supported in SQLite < 3.38.0",
)
@pytest.mark.broken(
    ["flink"],
    reason="https://github.com/ibis-project/ibis/pull/6920#discussion_r1373212503",
)
@pytest.mark.broken(
    ["risingwave"], reason="TODO(Kexiang): order mismatch in array", strict=False
)
def test_json_getitem_array(json_t):
    expr_fn = lambda t: t.js[1].name("res")
    expected = frozenset([None] * 4 + [47, None])
    expr = expr_fn(json_t)
    result = frozenset(expr.execute().replace({np.nan: None}))
    assert result == expected


@pytest.mark.notimpl(["dask", "mysql", "pandas", "risingwave"])
@pytest.mark.notyet(["bigquery", "sqlite"], reason="doesn't support maps")
@pytest.mark.notyet(["postgres"], reason="only supports map<string, string>")
@pytest.mark.notyet(
    ["pyspark", "trino", "flink"], reason="should work but doesn't deserialize JSON"
)
def test_json_map(backend, json_t):
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
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask", "mysql", "pandas", "risingwave"])
@pytest.mark.notyet(["sqlite"], reason="doesn't support arrays")
@pytest.mark.notyet(
    ["pyspark", "trino", "flink"], reason="should work but doesn't deserialize JSON"
)
@pytest.mark.notyet(["bigquery"], reason="doesn't allow null in arrays")
def test_json_array(backend, json_t):
    expr = json_t.js.array.name("res")
    result = expr.execute()
    expected = pd.Series(
        [None, None, None, None, [42, 47, 55], []], name="res", dtype="object"
    )
    backend.assert_series_equal(result, expected)
