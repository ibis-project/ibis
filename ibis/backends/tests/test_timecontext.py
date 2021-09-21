import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis.config import option_context

from .test_vectorized_udf import calc_mean, create_demean_struct_udf

GROUPBY_COL = 'month'
ORDERBY_COL = 'timestamp_col'
TARGET_COL = 'float_col'


@pytest.fixture
def context():
    # These need to be tz-naive because the timestamp_col in
    # the test data is tz-naive
    return pd.Timestamp('20090105'), pd.Timestamp('20090111')


def filter_by_time_context(df, context):
    return df[
        (df['timestamp_col'] >= context[0])
        & (df['timestamp_col'] < context[1])
    ]


@pytest.mark.only_on_backends(['pandas', 'pyspark'])
@pytest.mark.min_spark_version('3.1')
@pytest.mark.parametrize(
    'window',
    [
        ibis.trailing_window(ibis.interval(days=3), order_by=ORDERBY_COL),
        ibis.trailing_window(
            ibis.interval(days=3),
            order_by=ORDERBY_COL,
            group_by=GROUPBY_COL,
        ),
    ],
)
def test_context_adjustment_window_udf(alltypes, df, context, window):
    """This test case aims to test context adjustment of
    udfs in window method.
    """
    with option_context('context_adjustment.time_col', 'timestamp_col'):
        expr = alltypes.mutate(v1=calc_mean(alltypes[TARGET_COL]).over(window))
        result = expr.execute(timecontext=context)

        expected = expr.execute()
        expected = filter_by_time_context(expected, context).reset_index(
            drop=True
        )

        tm.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends(['pandas', 'pyspark'])
def test_context_adjustment_filter_before_window(alltypes, df, context):
    with option_context('context_adjustment.time_col', 'timestamp_col'):
        window = ibis.trailing_window(
            ibis.interval(days=3), order_by=ORDERBY_COL
        )

        expr = alltypes[alltypes['bool_col']]
        expr = expr.mutate(v1=expr[TARGET_COL].count().over(window))

        result = expr.execute(timecontext=context)

        expected = expr.execute()
        expected = filter_by_time_context(expected, context)
        expected = expected.reset_index(drop=True)

        tm.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends(['pandas'])
def test_context_adjustment_multi_col_udf_non_grouped(alltypes, df, context):
    with option_context('context_adjustment.time_col', 'timestamp_col'):
        w = ibis.window(preceding=None, following=None)

        demean_struct_udf = create_demean_struct_udf(
            result_formatter=lambda v1, v2: (v1, v2)
        )

        result = alltypes.mutate(
            demean_struct_udf(alltypes['double_col'], alltypes['int_col'])
            .over(w)
            .destructure()
        ).execute(timecontext=context)

        expected = alltypes.mutate(
            demean=alltypes['double_col']
            - alltypes['double_col'].mean().over(w),
            demean_weight=alltypes['int_col']
            - alltypes['int_col'].mean().over(w),
        ).execute(timecontext=context)
        tm.assert_frame_equal(result, expected)
