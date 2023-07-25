from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas import testing as tm
from pytest import param

from ibis.backends.pandas.aggcontext import Summarize, window_agg_udf

df = pd.DataFrame(
    {
        "id": [1, 2, 1, 2],
        "v1": [1.0, 2.0, 3.0, 4.0],
        "v2": [10.0, 20.0, 30.0, 40.0],
    }
)


@pytest.mark.parametrize(
    ("agg_fn", "expected_fn"),
    [
        param(
            lambda v1: v1.mean(),
            lambda df: df["v1"].mean(),
            id="udf",
        ),
        param(
            "mean",
            lambda df: df["v1"].mean(),
            id="string",
        ),
    ],
)
def test_summarize_single_series(agg_fn, expected_fn):
    """Test Summarize.agg operating on a single Series."""

    aggcontext = Summarize()

    result = aggcontext.agg(df["v1"], agg_fn)
    expected = expected_fn(df)

    assert result == expected


@pytest.mark.parametrize(
    ("agg_fn", "expected_fn"),
    [
        param(
            lambda v1: v1.mean(),
            lambda df: df["v1"].mean(),
            id="udf",
        ),
        param(
            "mean",
            lambda df: df["v1"].mean(),
            id="string",
        ),
    ],
)
def test_summarize_single_seriesgroupby(agg_fn, expected_fn):
    """Test Summarize.agg operating on a single SeriesGroupBy."""

    aggcontext = Summarize()

    df_grouped = df.sort_values("id").groupby("id")
    result = aggcontext.agg(df_grouped["v1"], agg_fn)

    expected = expected_fn(df_grouped)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("agg_fn", "expected_fn"),
    [
        param(
            lambda v1, v2: v1.mean() - v2.mean(),
            lambda df: df["v1"].mean() - df["v2"].mean(),
            id="two-column",
        ),
        # Two columns, but only the second one is actually used in UDF
        param(
            lambda v1, v2: v2.mean(),
            lambda df: df["v2"].mean(),
            id="redundant-column",
        ),
    ],
)
def test_summarize_multiple_series(agg_fn, expected_fn):
    """Test Summarize.agg operating on many Series."""

    aggcontext = Summarize()

    args = [df["v1"], df["v2"]]
    result = aggcontext.agg(args[0], agg_fn, *args[1:])

    expected = expected_fn(df)

    assert result == expected


@pytest.mark.parametrize(
    "param",
    [
        (
            pd.Series([True, True, True, True]),
            pd.Series([1.0, 2.0, 2.0, 3.0]),
        ),
        (
            pd.Series([False, True, True, False]),
            pd.Series([np.NaN, 2.0, 2.0, np.NaN]),
        ),
    ],
)
def test_window_agg_udf(param):
    """Test passing custom window indices for window aggregation."""

    mask, expected = param

    grouped_data = df.sort_values("id").groupby("id")["v1"]
    result_index = grouped_data.obj.index

    window_lower_indices = pd.Series([0, 0, 2, 2])
    window_upper_indices = pd.Series([1, 2, 3, 4])

    result = window_agg_udf(
        grouped_data,
        lambda s: s.mean(),
        window_lower_indices,
        window_upper_indices,
        mask,
        result_index,
        dtype="float",
        max_lookback=None,
    )

    expected.index = grouped_data.obj.index

    tm.assert_series_equal(result, expected)


def test_window_agg_udf_different_freq():
    """Test that window_agg_udf works when the window series and data series
    have different frequencies."""

    time = pd.Series([pd.Timestamp("20200101"), pd.Timestamp("20200201")])
    data = pd.Series([1, 2, 3, 4, 5, 6])
    window_lower_indices = pd.Series([0, 4])
    window_upper_indices = pd.Series([5, 7])
    mask = pd.Series([True, True])
    result_index = time.index

    result = window_agg_udf(
        data,
        lambda s: s.mean(),
        window_lower_indices,
        window_upper_indices,
        mask,
        result_index,
        "float",
        None,
    )

    expected = pd.Series([data.iloc[0:5].mean(), data.iloc[4:7].mean()])

    tm.assert_series_equal(result, expected)
