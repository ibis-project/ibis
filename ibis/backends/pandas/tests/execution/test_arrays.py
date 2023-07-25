from __future__ import annotations

import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

import ibis
from ibis.backends.pandas.tests.conftest import TestConf as tm


@pytest.mark.parametrize("arr", [[1, 3, 5], np.array([1, 3, 5])])
@pytest.mark.parametrize("create_arr_expr", [ibis.literal, ibis.array])
def test_array_literal(client, arr, create_arr_expr):
    expr = create_arr_expr(arr)
    result = client.execute(expr)
    expected = np.array([1, 3, 5])
    nt.assert_array_equal(result, expected)


def test_array_length(t):
    expr = t.select(
        t.array_of_float64.length().name("array_of_float64_length"),
        t.array_of_int64.length().name("array_of_int64_length"),
        t.array_of_strings.length().name("array_of_strings_length"),
    )
    result = expr.execute()
    expected = pd.DataFrame(
        {
            "array_of_float64_length": [2, 1, 0],
            "array_of_int64_length": [2, 0, 1],
            "array_of_strings_length": [2, 0, 1],
        }
    )

    tm.assert_frame_equal(result, expected)


def test_array_length_scalar(client):
    raw_value = np.array([1, 2, 4])
    value = ibis.array(raw_value)
    expr = value.length()
    result = client.execute(expr)
    expected = len(raw_value)
    assert result == expected


def test_array_collect(t, df):
    expr = t.float64_with_zeros.collect()
    result = expr.execute()
    expected = np.array(df.float64_with_zeros)
    nt.assert_array_equal(result, expected)


def test_array_collect_grouped(t, df):
    expr = t.group_by(t.dup_strings).aggregate(collected=t.float64_with_zeros.collect())
    result = expr.execute().sort_values("dup_strings").reset_index(drop=True)
    expected = (
        df.groupby("dup_strings")
        .float64_with_zeros.apply(np.array)
        .reset_index()
        .rename(columns={"float64_with_zeros": "collected"})
    )
    tm.assert_frame_equal(result, expected)


def test_array_collect_rolling_partitioned(t, df):
    window = ibis.trailing_window(1, order_by=t.plain_int64)
    colexpr = t.plain_float64.collect().over(window)
    expr = t["dup_strings", "plain_int64", colexpr.name("collected")]
    result = expr.execute()
    expected = pd.DataFrame(
        {
            "dup_strings": ["d", "a", "d"],
            "plain_int64": [1, 2, 3],
            "collected": [[4.0], [4.0, 5.0], [5.0, 6.0]],
        }
    )[expr.columns]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ["start", "stop"],
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        (-3, None),
        (None, -3),
        (-3, -1),
    ],
)
def test_array_slice(t, df, start, stop):
    expr = t.array_of_strings[start:stop]
    result = expr.execute()
    expected = df.array_of_strings.apply(lambda x: x[start:stop].tolist())
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ["start", "stop"],
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        (-3, None),
        (None, -3),
        (-3, -1),
    ],
)
def test_array_slice_scalar(client, start, stop):
    raw_value = np.array([-11, 42, 10])
    value = ibis.array(raw_value)
    expr = value[start:stop]
    result = client.execute(expr)
    expected = raw_value[start:stop]
    nt.assert_array_equal(result, expected)


@pytest.mark.parametrize("index", [1, 3, 4, 11, -11])
def test_array_index(t, df, index):
    expr = t[t.array_of_float64[index].name("indexed")]
    result = expr.execute()
    expected = pd.DataFrame(
        {
            "indexed": df.array_of_float64.apply(
                lambda x: x[index] if -len(x) <= index < len(x) else np.nan
            )
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index", [1, 3, 4, 11])
def test_array_index_scalar(client, index):
    raw_value = np.array([-10, 1, 2, 42])
    value = ibis.array(raw_value)
    expr = value[index]
    result = client.execute(expr)
    expected = raw_value[index] if index < len(raw_value) else None
    assert result == expected


@pytest.mark.parametrize("n", [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize("mul", [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat(t, df, n, mul):
    expr = mul(t.array_of_strings, n)
    result = expr.execute()
    expected = df.apply(
        lambda row: np.tile(row.array_of_strings, max(n, 0)).tolist(),
        axis=1,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("n", [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize("mul", [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat_scalar(client, n, mul):
    raw_array = np.array([1, 2])
    array = ibis.array(raw_array)
    expr = mul(array, n)
    result = client.execute(expr)
    if n > 0:
        expected = np.tile(raw_array, n)
    else:
        expected = np.array([], dtype=raw_array.dtype)
    nt.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ["op", "op_raw"],
    [
        (lambda x, y: x + y, lambda x, y: np.concatenate([x, y])),
        (lambda x, y: y + x, lambda x, y: np.concatenate([y, x])),
    ],
)
def test_array_concat(t, df, op, op_raw):
    x = t.array_of_float64.cast("array<string>")
    y = t.array_of_strings
    expr = op(x, y)
    result = expr.execute()
    expected = df.apply(
        lambda row: op_raw(
            np.array(list(map(str, row.array_of_float64))),  # Mimic .cast()
            row.array_of_strings,
        ),
        axis=1,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ["op", "op_raw"],
    [
        (lambda x, y: x + y, lambda x, y: np.concatenate([x, y])),
        (lambda x, y: y + x, lambda x, y: np.concatenate([y, x])),
    ],
)
def test_array_concat_scalar(client, op, op_raw):
    raw_left = np.array([1, 2, 3])
    raw_right = np.array([3, 4])
    left = ibis.array(raw_left)
    right = ibis.array(raw_right)
    expr = op(left, right)
    result = client.execute(expr)
    expected = op_raw(raw_left, raw_right)
    nt.assert_array_equal(result, expected)
