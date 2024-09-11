"""Tests for json data types."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir


@pytest.fixture(scope="module")
def jsonb_t(con):
    return con.table("jsonb_t")


@pytest.mark.parametrize("data", [param({"status": True}, id="status")])
def test_json(data, alltypes):
    lit = ibis.literal(json.dumps(data), type="json").name("tmp")
    expr = alltypes.select(alltypes.id, lit).head(1)
    df = expr.execute()
    assert df["tmp"].iloc[0] == data


def test_jsonb_extract_path(con):
    json_t = con.table("json_t")
    jsonb_t = con.table("jsonb_t")

    assert json_t.js.type() == dt.JSON(binary=False)
    assert jsonb_t.js.type() == dt.JSON(binary=True)

    tm.assert_series_equal(jsonb_t.js["a"].execute(), json_t.js["a"].execute())


def test_json_getitem_object(jsonb_t):
    expr_fn = lambda t: t.js["a"].name("res")
    expected = frozenset([(1, 2, 3, 4), None, "foo"] + [None] * 3)
    expr = expr_fn(jsonb_t)
    result = frozenset(
        expr.execute()
        .map(lambda o: tuple(o) if isinstance(o, list) else o)
        .replace({np.nan: None})
    )
    assert result == expected


def test_json_getitem_array(jsonb_t):
    expr_fn = lambda t: t.js[1].name("res")
    expected = frozenset([None] * 4 + [47, None])
    expr = expr_fn(jsonb_t)
    result = frozenset(expr.execute().replace({np.nan: None}))
    assert result == expected


def test_json_array(jsonb_t):
    expr = jsonb_t.mutate("rowid", res=jsonb_t.js.array).order_by("rowid")
    result = expr.execute().res
    expected = pd.Series(
        [None, None, None, None, [42, 47, 55], []] + [None] * 8,
        name="res",
        dtype="object",
    )
    tm.assert_series_equal(result, expected)


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
def test_json_unwrap(jsonb_t, typ, expected_data, expr_fn):
    expr = expr_fn(jsonb_t.js, typ).name("res")
    result = expr.execute()
    expected = pd.Series(expected_data, name="res", dtype="object")
    tm.assert_series_equal(
        result.replace(np.nan, None).fillna(pd.NA).sort_values().reset_index(drop=True),
        expected.replace(np.nan, None)
        .fillna(pd.NA)
        .sort_values()
        .reset_index(drop=True),
        check_dtype=False,
    )
