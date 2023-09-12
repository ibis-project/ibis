from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis

dd = pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402


def test_array_length(t):
    expr = t.select(
        t.array_of_float64.length().name("array_of_float64_length"),
        t.array_of_int64.length().name("array_of_int64_length"),
        t.array_of_strings.length().name("array_of_strings_length"),
    )
    result = expr.compile()
    expected = dd.from_pandas(
        pd.DataFrame(
            {
                "array_of_float64_length": [2, 1, 0],
                "array_of_int64_length": [2, 0, 1],
                "array_of_strings_length": [2, 0, 1],
            }
        ),
        npartitions=1,
    )

    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


def test_array_length_scalar(client):
    raw_value = [1, 2, 4]
    value = ibis.literal(raw_value)
    expr = value.length()
    result = client.execute(expr)
    expected = len(raw_value)
    assert result == expected


def test_array_collect(t, df):
    expr = t.group_by(t.dup_strings).aggregate(collected=t.float64_with_zeros.collect())
    result = expr.compile()
    expected = (
        df.groupby("dup_strings")
        .float64_with_zeros.apply(list)
        .reset_index()
        .rename(columns={"float64_with_zeros": "collected"})
    )
    tm.assert_frame_equal(
        result.compute().sort_values(["dup_strings"]).reset_index(drop=True),
        expected.compute().sort_values(["dup_strings"]).reset_index(drop=True),
    )


@pytest.mark.notimpl(["dask"], reason="windowing - #2553")
def test_array_collect_rolling_partitioned(t, df):
    window = ibis.trailing_window(1, order_by=t.plain_int64)
    colexpr = t.plain_float64.collect().over(window)
    expr = t["dup_strings", "plain_int64", colexpr.name("collected")]
    result = expr.compile()
    expected = dd.from_pandas(
        pd.DataFrame(
            {
                "dup_strings": ["d", "a", "d"],
                "plain_int64": [1, 2, 3],
                "collected": [[4.0], [4.0, 5.0], [5.0, 6.0]],
            }
        ),
        npartitions=1,
    )[expr.columns]
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


# Need an ops.ArraySlice execution func that dispatches on dd.Series
@pytest.mark.notimpl(["dask"], reason="arrays - #2553")
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
    result = expr.compile()
    slicer = operator.itemgetter(slice(start, stop))
    expected = df.array_of_strings.apply(slicer)
    tm.assert_series_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


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
    raw_value = [-11, 42, 10]
    value = ibis.literal(raw_value)
    expr = value[start:stop]
    result = client.execute(expr)
    expected = raw_value[start:stop]
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [param(1, marks=pytest.mark.xfail_version(dask=["pandas>=2"])), 3, 4, 11, -11],
)
def test_array_index(t, df, index):
    expr = t[t.array_of_float64[index].name("indexed")]
    result = expr.compile()
    expected = dd.from_pandas(
        pd.DataFrame(
            {
                "indexed": df.array_of_float64.apply(
                    lambda x: x[index] if -len(x) <= index < len(x) else None,
                    meta=("array_of_float64", "object"),
                )
            }
        ),
        npartitions=1,
    )
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize("index", [1, 3, 4, 11])
def test_array_index_scalar(client, index):
    raw_value = [-10, 1, 2, 42]
    value = ibis.literal(raw_value)
    expr = value[index]
    result = client.execute(expr)
    expected = raw_value[index] if index < len(raw_value) else None
    assert result == expected


@pytest.mark.notimpl(["dask"], reason="arrays - #2553")
@pytest.mark.parametrize("n", [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize("mul", [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat(t, df, n, mul):
    expr = t.select(repeated=mul(t.array_of_strings, n))
    result = expr.execute()
    expected = pd.DataFrame({"repeated": df.array_of_strings * n})
    tm.assert_frame_equal(result, expected)


# ValueError: Dask backend borrows Pandas backend's Cast execution
# function, which assumes array representation is np.array.
# NotImplementedError: Need an ops.ArrayConcat execution func that
# dispatches on dd.Series
@pytest.mark.notimpl(["dask"], reason="arrays - #2553")
@pytest.mark.parametrize("op", [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat(t, df, op):
    x = t.array_of_float64.cast("array<string>")
    y = t.array_of_strings
    expr = op(x, y)
    result = expr.compile()
    expected = op(
        df.array_of_float64.apply(lambda x: list(map(str, x))),
        df.array_of_strings,
    )
    tm.assert_series_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


@pytest.mark.parametrize("op", [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat_scalar(client, op):
    raw_left = [1, 2, 3]
    raw_right = [3, 4]
    left = ibis.literal(raw_left)
    right = ibis.literal(raw_right)
    expr = op(left, right)
    result = client.execute(expr)
    expected = op(raw_left, raw_right)
    assert np.array_equal(result, expected)
