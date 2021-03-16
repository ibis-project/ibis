import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.config import option_context
from ibis.udf.vectorized import reduction

GROUPBY_COL = 'month'
ORDERBY_COL = 'timestamp_col'
TARGET_COL = 'float_col'


@reduction(input_type=[dt.double], output_type=dt.double)
def calc_mean(series):
    return series.mean()


@pytest.fixture
def df(alltypes):
    return alltypes.execute().sort_values(by=[ORDERBY_COL])


@pytest.fixture
def context():
    return pd.Timestamp('20090105'), pd.Timestamp('20090111')


@pytest.fixture
def expected_series(df, context):
    # expected result series for context adjustment tests
    exp_raw_win = df.set_index(ORDERBY_COL).assign(v1=df[TARGET_COL].mean())
    exp_raw_win = exp_raw_win[exp_raw_win.index >= context[0]]
    exp_raw_win = exp_raw_win[exp_raw_win.index <= context[1]].reset_index(
        drop=True
    )['v1']

    exp_orderby = (
        df.set_index(ORDERBY_COL)
        .rename(columns={TARGET_COL: 'v1'})['v1']
        .rolling('3d', closed='both')
        .mean()
    )
    exp_orderby = exp_orderby[exp_orderby.index >= context[0]]
    exp_orderby = exp_orderby[exp_orderby.index <= context[1]].reset_index(
        drop=True
    )

    exp_groupby_orderby = (
        df.set_index(ORDERBY_COL)
        .groupby(GROUPBY_COL)
        .rolling('3d', closed='both')[TARGET_COL]
        .mean()
    ).reset_index()

    # Result is a MultiIndexed Series
    exp_groupby_orderby = exp_groupby_orderby[
        exp_groupby_orderby[ORDERBY_COL] >= context[0]
    ]
    exp_groupby_orderby = exp_groupby_orderby[
        exp_groupby_orderby[ORDERBY_COL] <= context[1]
    ]
    exp_groupby_orderby = exp_groupby_orderby.reset_index(drop=True)[
        TARGET_COL
    ].rename('v1')
    return [exp_raw_win, exp_orderby, exp_groupby_orderby]


@pytest.mark.only_on_backends(['pandas', 'pyspark'])
@pytest.mark.xfail_unsupported
@pytest.mark.parametrize(
    ['window', 'exp_idx'],
    [
        (ibis.trailing_window(3 * ibis.interval(days=1)), 0),
        (
            ibis.trailing_window(
                3 * ibis.interval(days=1), order_by=ORDERBY_COL
            ),
            1,
        ),
        (
            ibis.trailing_window(
                3 * ibis.interval(days=1),
                order_by=ORDERBY_COL,
                group_by=GROUPBY_COL,
            ),
            2,
        ),
    ],
)
def test_context_adjustment_window_udf(
    alltypes, df, context, expected_series, window, exp_idx
):
    """ This test case aims to test context adjustment of
        udfs in window method.
    """
    with option_context('context_adjustment.time_col', 'timestamp_col'):
        expr = alltypes.mutate(v1=calc_mean(alltypes[TARGET_COL]).over(window))
        result = expr.execute(timecontext=context)
        tm.assert_series_equal(result["v1"], expected_series[exp_idx])
