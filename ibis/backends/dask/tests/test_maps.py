from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ibis

dd = pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402


def test_map_length_expr(t):
    expr = t.map_of_integers_strings.length()
    result = expr.execute()
    expected = pd.Series([0, None, 2], name="MapLength(map_of_integers_strings)")
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_value_for_key_expr(t):
    expr = t.map_of_integers_strings[1]
    result = expr.execute()
    expected = pd.Series(
        [None, None, "a"], name="MapGet(map_of_integers_strings, 1, None)"
    )
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_value_or_default_for_key_expr(t):
    expr = t.map_of_complex_values.get("a")
    result = expr.execute()
    expected = pd.Series(
        [None, [1, 2, 3], None],
        dtype="object",
        name=expr.get_name(),
    )
    tm.assert_series_equal(result, expected, check_index=False)


def safe_sorter(element):
    return sorted(element) if isinstance(element, list) else element


def test_map_keys_expr(t):
    expr = t.map_of_strings_integers.keys()
    result = expr.execute().map(safe_sorter)
    expected = pd.Series(
        [["a", "b"], None, []],
        dtype="object",
        name="MapKeys(map_of_strings_integers)",
    ).map(safe_sorter)
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_values_expr(t):
    expr = t.map_of_complex_values.values()
    result = expr.execute()
    expected = pd.Series(
        [
            None,
            np.array([[1, 2, 3], []], dtype="object"),
            np.array([], dtype="object"),
        ],
        dtype="object",
        name="MapValues(map_of_complex_values)",
    )
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_concat_expr(t):
    expr = t.map_of_complex_values + {"b": [4, 5, 6], "c": [], "a": []}
    result = expr.execute()
    expected = pd.Series(
        [
            None,
            {"a": [], "b": [4, 5, 6], "c": []},
            {"b": [4, 5, 6], "c": [], "a": []},
        ],
        dtype="object",
        name=expr.get_name(),
    )
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_value_for_key_literal_broadcast(t):
    lookup_table = ibis.literal({"a": 1, "b": 2, "c": 3, "d": 4})
    expr = lookup_table.get(t.dup_strings)
    result = expr.execute()
    expected = pd.Series([4, 1, 4], dtype="int8", name=expr.get_name())
    tm.assert_series_equal(result, expected, check_index=False)
