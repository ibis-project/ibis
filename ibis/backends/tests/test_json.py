"""Tests for JSON operations."""

from __future__ import annotations

import sqlite3

import pytest
from packaging.version import parse as vparse

import ibis.expr.types as ir
from ibis.backends.tests.conftest import NO_JSON_SUPPORT_MARKS
from ibis.backends.tests.errors import PySparkPythonException
from ibis.conftest import IS_SPARK_REMOTE

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = NO_JSON_SUPPORT_MARKS


@pytest.mark.notyet(
    ["sqlite"],
    condition=vparse(sqlite3.sqlite_version) < vparse("3.38.0"),
    reason="JSON not supported in SQLite < 3.38.0",
)
@pytest.mark.notimpl(
    ["flink"],
    reason="https://github.com/ibis-project/ibis/pull/6920#discussion_r1373212503",
)
@pytest.mark.notimpl(
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
@pytest.mark.notimpl(
    ["flink"],
    reason="https://github.com/ibis-project/ibis/pull/6920#discussion_r1373212503",
)
@pytest.mark.notimpl(
    ["risingwave"], reason="TODO(Kexiang): order mismatch in array", strict=False
)
def test_json_getitem_array(json_t):
    expr_fn = lambda t: t.js[1].name("res")
    expected = frozenset([None] * 4 + [47, None])
    expr = expr_fn(json_t)
    result = frozenset(expr.execute().replace({np.nan: None}))
    assert result == expected


@pytest.mark.notimpl(["mysql", "risingwave"])
@pytest.mark.notyet(["bigquery", "sqlite"], reason="doesn't support maps")
@pytest.mark.notyet(["postgres"], reason="only supports map<string, string>")
@pytest.mark.notyet(
    ["pyspark", "flink"], reason="should work but doesn't deserialize JSON"
)
def test_json_map(backend, json_t):
    expr = json_t.mutate("rowid", res=json_t.js.map).order_by("rowid")
    result = expr.execute().res
    expected = pd.Series(
        [
            {"a": [1, 2, 3, 4], "b": 1},
            {"a": None, "b": 2},
            {"a": "foo", "c": None},
        ]
        + [None] * 11,
        dtype="object",
        name="res",
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["mysql", "risingwave"])
@pytest.mark.notyet(["sqlite"], reason="doesn't support arrays")
@pytest.mark.notyet(
    ["pyspark", "flink"], reason="should work but doesn't deserialize JSON"
)
@pytest.mark.notyet(["bigquery"], reason="doesn't allow null in arrays")
def test_json_array(backend, json_t):
    expr = json_t.mutate("rowid", res=json_t.js.array).order_by("rowid")
    result = expr.execute().res
    expected = pd.Series(
        [None, None, None, None, [42, 47, 55], []] + [None] * 8,
        name="res",
        dtype="object",
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.notyet(
    ["sqlite"],
    condition=vparse(sqlite3.sqlite_version) < vparse("3.38.0"),
    reason="JSON not supported in SQLite < 3.38.0",
)
@pytest.mark.notimpl(["risingwave"])
@pytest.mark.notyet(["flink"], reason="should work but doesn't deserialize JSON")
@pytest.mark.notyet(
    ["pyspark"],
    condition=IS_SPARK_REMOTE,
    raises=PySparkPythonException,
    reason="environment issues",
)
@pytest.mark.parametrize(
    ("typ", "expected_data"),
    [
        ("str", [None] * 6 + ["a", "", "b"] + [None] * 5),
        ("int", [None] * 12 + [42, None]),
        ("float", [None] * 12 + [42.0, 37.37]),
        ("bool", [None] * 10 + [True, False, None, None]),
    ],
    ids=["str", "int", "float", "bool"],
)
@pytest.mark.parametrize(
    "expr_fn", [getattr, ir.JSONValue.unwrap_as], ids=["getattr", "unwrap_as"]
)
def test_json_unwrap(backend, json_t, typ, expected_data, expr_fn):
    expr = expr_fn(json_t.js, typ).name("res")
    result = expr.execute()
    expected = pd.Series(expected_data, name="res", dtype="object")
    backend.assert_series_equal(
        result.replace(np.nan, None).fillna(pd.NA).sort_values().reset_index(drop=True),
        expected.replace(np.nan, None)
        .fillna(pd.NA)
        .sort_values()
        .reset_index(drop=True),
        check_dtype=False,
    )
