from __future__ import annotations

import pandas as pd
import pytest

import ibis

dd = pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402


def test_map_length_expr(t):
    expr = t.map_of_integers_strings.length()
    result = expr.execute()
    expected = pd.Series(
        [2, 0, None], dtype="object", name="MapLength(map_of_integers_strings)"
    )
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_value_for_key_expr(t):
    expr = t.map_of_integers_strings[1]
    result = expr.execute()
    expected = pd.Series(
        ["a", None, None], name="MapGet(map_of_integers_strings, 1, None)"
    )
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_value_or_default_for_key_expr(t):
    expr = t.map_of_complex_values.get("a")
    result = expr.execute()
    expected = pd.Series(
        [[1, 2, 3], None, None],
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
        [["a", "b"], [], None],
        dtype="object",
        name="MapKeys(map_of_strings_integers)",
    ).map(safe_sorter)
    tm.assert_series_equal(result, expected, check_index=False)


def test_map_values_expr(t):
    expr = t.map_of_complex_values.values()
    result = list(expr.execute())
    expected = [
        [[1, 2, 3], []],
        [],
        None,
    ]
    assert result == expected


def test_map_concat_expr(t):
    expr = t.map_of_complex_values + {"b": [4, 5, 6], "c": [], "a": []}
    result = expr.execute()
    expected = pd.Series(
        [
            {"a": [], "b": [4, 5, 6], "c": []},
            {"a": [], "b": [4, 5, 6], "c": []},
            None,
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
