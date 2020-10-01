import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as tm

from ..aggcontext import window_agg_udf


@pytest.mark.parametrize(
    'param',
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
    """ Test passing custom window indices for window aggregation."""

    mask, expected = param

    df = pd.DataFrame({'id': [1, 2, 1, 2], 'v': [1.0, 2.0, 3.0, 4.0]})

    grouped_data = df.sort_values('id').groupby("id")['v']
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
        dtype='float',
        max_lookback=None,
    )

    expected.index = grouped_data.obj.index

    tm.assert_series_equal(result, expected)


def test_window_agg_udf_different_freq():
    """ Test that window_agg_udf works when the window series and data series
    have different frequencies.
    """

    time = pd.Series([pd.Timestamp('20200101'), pd.Timestamp('20200201')])
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
        'float',
        None,
    )

    expected = pd.Series([data.iloc[0:5].mean(), data.iloc[4:7].mean()])

    tm.assert_series_equal(result, expected)
