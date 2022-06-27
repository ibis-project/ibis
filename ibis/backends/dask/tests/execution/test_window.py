# import io
# from datetime import date
# from operator import methodcaller

# import numpy as np
import pandas as pd
import pytest
from dask.dataframe.utils import tm

import ibis

# import ibis.common.exceptions as com
# import ibis.expr.datatypes as dt
# import ibis.expr.operations as ops
# from ibis.backends.dask import Backend
from ibis.backends.dask.execution import execute
from ibis.backends.pandas.aggcontext import AggregationContext, window_agg_udf

# from packaging.version import parse as vparse


# from ibis.expr.scope import Scope
# from ibis.expr.window import get_preceding_value, rows_with_max_lookback
# from ibis.udf.vectorized import reduction

# These custom classes are used inn test_custom_window_udf


class CustomInterval:
    def __init__(self, value):
        self.value = value

    # These are necessary because ibis.expr.window
    # will compare preceding and following
    # with 0 to see if they are valid
    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other


class CustomWindow(ibis.expr.window.Window):
    """This is a dummy custom window that return n preceding rows
    where n is defined by CustomInterval.value."""

    def _replace(self, **kwds):
        new_kwds = {
            'group_by': kwds.get('group_by', self._group_by),
            'order_by': kwds.get('order_by', self._order_by),
            'preceding': kwds.get('preceding', self.preceding),
            'following': kwds.get('following', self.following),
            'max_lookback': kwds.get('max_lookback', self.max_lookback),
            'how': kwds.get('how', self.how),
        }
        return CustomWindow(**new_kwds)


class CustomAggContext(AggregationContext):
    def __init__(
        self, parent, group_by, order_by, output_type, max_lookback, preceding
    ):
        super().__init__(
            parent=parent,
            group_by=group_by,
            order_by=order_by,
            output_type=output_type,
            max_lookback=max_lookback,
        )
        self.preceding = preceding

    def agg(self, grouped_data, function, *args, **kwargs):
        upper_indices = pd.Series(range(1, len(self.parent) + 2))
        window_sizes = (
            grouped_data.rolling(self.preceding.value + 1, min_periods=0)
            .count()
            .reset_index(drop=True)
        )
        lower_indices = upper_indices - window_sizes
        mask = upper_indices.notna()

        result_index = grouped_data.obj.index

        result = window_agg_udf(
            grouped_data,
            function,
            lower_indices,
            upper_indices,
            mask,
            result_index,
            self.dtype,
            self.max_lookback,
            *args,
            **kwargs,
        )

        return result


@pytest.fixture(scope='session')
def sort_kind():
    return 'mergesort'


default = pytest.mark.parametrize('default', [ibis.NA, ibis.literal('a')])
row_offset = pytest.mark.parametrize(
    'row_offset', list(map(ibis.literal, [-1, 1, 0]))
)
range_offset = pytest.mark.parametrize(
    'range_offset',
    [
        ibis.interval(days=1),
        2 * ibis.interval(days=1),
        -2 * ibis.interval(days=1),
    ],
)


@pytest.fixture
def row_window():
    return ibis.window(following=0, order_by='plain_int64')


@pytest.fixture
def range_window():
    return ibis.window(following=0, order_by='plain_datetimes_naive')


@pytest.fixture
def custom_window():
    return CustomWindow(
        preceding=CustomInterval(1),
        following=0,
        group_by='dup_ints',
        order_by='plain_int64',
    )


@default
@row_offset
def test_lead(t, df, row_offset, default, row_window):
    expr = t.dup_strings.lead(row_offset, default=default).over(row_window)
    result = expr.compile()
    expected = df.dup_strings.shift(execute(-row_offset))
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result.compute(), expected.compute())


@default
@row_offset
def test_lag(t, df, row_offset, default, row_window):
    expr = t.dup_strings.lag(row_offset, default=default).over(row_window)
    result = expr.compile()
    expected = df.dup_strings.shift(execute(row_offset))
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result.compute(), expected.compute())


@default
@range_offset
def test_lag_delta(t, df, range_offset, default, range_window):
    expr = t.dup_strings.lag(range_offset, default=default).over(range_window)
    result = expr.compile()
    expected = (
        df[['plain_datetimes_naive', 'dup_strings']]
        .set_index('plain_datetimes_naive')
        .squeeze()
        .shift(freq=execute(range_offset))
        .reset_index(drop=True)
    )
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result.compute(), expected.compute())


def test_first(t, df):
    expr = t.dup_strings.first()
    result = expr.compile()
    assert result.compute() == df.dup_strings.compute().iloc[0]


def test_last(t, df):
    expr = t.dup_strings.last()
    result = expr.compile()
    assert result.compute() == df.dup_strings.compute().iloc[-1]
