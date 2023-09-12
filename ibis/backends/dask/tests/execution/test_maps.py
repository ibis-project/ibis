from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ibis

dd = pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402


def test_map_length_expr(t):
    expr = t.map_of_integers_strings.length()
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series([0, None, 2], name="map_of_integers_strings"),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_map_value_for_key_expr(t):
    expr = t.map_of_integers_strings[1]
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series([None, None, "a"], name="map_of_integers_strings"),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_map_value_or_default_for_key_expr(t):
    expr = t.map_of_complex_values.get("a")
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series(
            [None, [1, 2, 3], None],
            dtype="object",
            name="map_of_complex_values",
        ),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def safe_sorter(element):
    return sorted(element) if isinstance(element, list) else element


def test_map_keys_expr(t):
    expr = t.map_of_strings_integers.keys()
    result = expr.compile().map(safe_sorter)
    expected = dd.from_pandas(
        pd.Series(
            [["a", "b"], None, []],
            dtype="object",
            name="map_of_strings_integers",
        ),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_map_values_expr(t):
    expr = t.map_of_complex_values.values()
    result = expr.compile().map(safe_sorter)
    expected = dd.from_pandas(
        pd.Series(
            [
                None,
                np.array([[1, 2, 3], []], dtype="object"),
                np.array([], dtype="object"),
            ],
            dtype="object",
            name="map_of_complex_values",
        ),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_map_concat_expr(t):
    expr = t.map_of_complex_values + {"b": [4, 5, 6], "c": [], "a": []}
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series(
            [
                None,
                {"a": [], "b": [4, 5, 6], "c": []},
                {"b": [4, 5, 6], "c": [], "a": []},
            ],
            dtype="object",
            name="map_of_complex_values",
        ),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_map_value_for_key_literal_broadcast(t):
    lookup_table = ibis.literal({"a": 1, "b": 2, "c": 3, "d": 4})
    expr = lookup_table.get(t.dup_strings)
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series([4, 1, 4], name="dup_strings"),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)
