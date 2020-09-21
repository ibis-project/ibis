import dask.dataframe as dd
import numpy as np
import pandas as pd

import pytest
from dask.dataframe.utils import tm

from ibis.dask.aggcontext import window_agg_udf


@pytest.mark.parametrize(
    'param',
    [
        (
            dd.Series([True, True, True, True]),
            dd.Series([1.0, 2.0, 2.0, 3.0]),
        ),
        (
            dd.Series([False, True, True, False]),
            dd.Series([np.NaN, 2.0, 2.0, np.NaN]),
        ),
    ],
)
def test_window_agg_udf(param):
    """ Test passing custom window indices for window aggregation."""

    mask, expected = param

    df = dd.DataFrame({'id': [1, 2, 1, 2], 'v': [1.0, 2.0, 3.0, 4.0]})

    grouped_data = df.sort_values('id').groupby("id")['v']
    result_index = grouped_data.obj.index

    window_lower_indices = dd.Series([0, 0, 2, 2])
    window_upper_indices = dd.Series([1, 2, 3, 4])

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

    time = dd.Series([pd.Timestamp('20200101'), pd.Timestamp('20200201')])
    data = dd.Series([1, 2, 3, 4, 5, 6])
    window_lower_indices = dd.Series([0, 4])
    window_upper_indices = dd.Series([5, 7])
    mask = dd.Series([True, True])
    result_index = time.index

    result = window_agg_udf(
        time,
        lambda s: s.mean(),
        window_lower_indices,
        window_upper_indices,
        mask,
        result_index,
        'float',
        None,
        data,
    )

    expected = dd.Series([data.iloc[0:5].mean(), data.iloc[4:7].mean()])

    tm.assert_series_equal(result, expected)
