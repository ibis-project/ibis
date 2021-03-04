import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.udf.vectorized import reduction


@reduction(input_type=[dt.double], output_type=dt.double)
def calc_mean(series):
    return series.mean()


@pytest.fixture
def time_table(con, temp_table):
    df = pd.DataFrame(
        {
            'time': pd.Series(
                pd.date_range(
                    start='2017-01-02 01:02:03.234', periods=8
                ).values
            ),
            'id': list(range(1, 5)) * 2,
            'value': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
        }
    )
    df_schema = ibis.schema(
        [('time', 'timestamp'), ('id', 'int32'), ('value', 'float64')]
    )
    con.create_table(temp_table, schema=df_schema)
    con.load_data(temp_table, df, if_exists='append')
    return con.table(temp_table)


@pytest.fixture
def df(time_table):
    return time_table.execute()


@pytest.fixture
def context():
    return pd.Timestamp('20170105'), pd.Timestamp('20170111')


@pytest.fixture
def expected_series(df):
    # expected result series for context adjustment tests
    exp_raw_win = df.set_index('time').assign(v1=df['value'].mean())
    exp_raw_win = exp_raw_win[
        exp_raw_win.index >= pd.Timestamp('20170105')
    ].reset_index(drop=True)['v1']

    exp_orderby = (
        df.set_index('time')
        .rename(columns={'value': 'v1'})['v1']
        .rolling('3d', closed='both')
        .mean()
    )
    exp_orderby = exp_orderby[
        exp_orderby.index >= pd.Timestamp('20170105')
    ].reset_index(drop=True)

    exp_groupby_orderby = (
        df.set_index('time').groupby('id').rolling('3d', closed='both').mean()
    )['value'].reset_index()
    # Result is a MultiIndexed Series
    exp_groupby_orderby = (
        exp_groupby_orderby[
            exp_groupby_orderby.time >= pd.Timestamp('20170105')
        ]
        .reset_index(drop=True)['value']
        .rename('v1')
    )
    return [exp_raw_win, exp_orderby, exp_groupby_orderby]


@pytest.mark.only_on_backends(['pandas', 'pyspark'])
@pytest.mark.xfail_unsupported
@pytest.mark.parametrize(
    ['window', 'exp_idx'],
    [
        (ibis.trailing_window(3 * ibis.interval(days=1)), 0),
        (ibis.trailing_window(3 * ibis.interval(days=1), order_by='time'), 1),
        (
            ibis.trailing_window(
                3 * ibis.interval(days=1), order_by='time', group_by='id'
            ),
            2,
        ),
    ],
)
def test_context_adjustment_window_udf(
    time_table, df, context, expected_series, window, exp_idx
):
    """ This test case aims to test context adjustment of
        udfs in window method.
    """
    expr = time_table.mutate(v1=calc_mean(time_table['value']).over(window))
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result["v1"], expected_series[exp_idx])
