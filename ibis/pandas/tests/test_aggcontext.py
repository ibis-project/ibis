import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as tm

from ibis.pandas.aggcontext import window_agg_udf


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

    window_lower_indices = pd.Series([0, 0, 2, 2])
    window_higher_indices = pd.Series([1, 2, 3, 4])

    result = window_agg_udf(
        grouped_data,
        lambda s: s.mean(),
        window_lower_indices,
        window_higher_indices,
        mask,
        dtype='float',
        max_lookback=None,
    )

    expected.index = grouped_data.obj.index

    tm.assert_series_equal(result, expected)
