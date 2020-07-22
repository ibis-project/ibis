import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis
import ibis.common.exceptions as com

pytestmark = pytest.mark.pandas


def test_execute_with_timecontext(time_table):
    expr = time_table
    # define a time context for time-series data
    context = ('20170101', '20170103')

    # without time context, execute produces every row
    df_all = expr.execute()
    assert len(df_all['time']) == 8

    # with context set, execute produces only rows within context
    df_within_context = expr.execute(timecontext=context)
    assert len(df_within_context['time']) == 1


merge_asof_minversion = pytest.mark.skipif(
    pd.__version__ < '0.19.2',
    reason='at least pandas-0.19.2 required for merge_asof',
)


def test_bad_timecontext(time_table):
    expr = time_table
    # define context with illegal string
    with pytest.raises(ValueError):
        context = ('bad', 'context')
        expr.execute(timecontext=context)
    # define context with unsupport type int
    with pytest.raises(com.IbisError):
        context = (20091010, 20100101)
        expr.execute(timecontext=context)
    # define context with too many values
    with pytest.raises(com.IbisError):
        context = ('20101010', '20100101', '20101010')
        expr.execute(timecontext=context)
    # define context with too few values
    with pytest.raises(com.IbisError):
        context = pd.Timestamp('20101010')
        expr.execute(timecontext=context)
    # define context with begin value later than end
    with pytest.raises(com.IbisError):
        context = ('20101010', '20090101')
        expr.execute(timecontext=context)


@merge_asof_minversion
def test_context_adjustment_asof_join(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(
        time_keyed_right, 'time', by='key', tolerance=4 * ibis.interval(days=1)
    )[time_keyed_left, time_keyed_right.right_value]
    context = (pd.Timestamp('20170105'), pd.Timestamp('20170111'))
    result = expr.execute(timecontext=context)

    # compare with asof_join of manually trimmed tables
    trimmed_df1 = time_keyed_df1[time_keyed_df1['time'] >= context[0]][
        time_keyed_df1['time'] < context[1]
    ]
    trimmed_df2 = time_keyed_df2[
        time_keyed_df2['time'] >= context[0] - pd.Timedelta(days=4)
    ][time_keyed_df2['time'] < context[1]]
    expected = pd.merge_asof(
        trimmed_df1,
        trimmed_df2,
        on='time',
        by='key',
        tolerance=pd.Timedelta('4D'),
    )
    tm.assert_frame_equal(result, expected)


def test_context_adjustment_window(time_table, time_df3):
    # trim data manually
    expected = (
        time_df3.set_index('time').value.rolling('3d', closed='both').mean()
    )
    expected = expected[
        expected.index >= pd.Timestamp('20170105')
    ].reset_index(drop=True)

    context = pd.Timestamp('20170105'), pd.Timestamp('20170111')

    # expected.index.name = None
    window = ibis.trailing_window(
        3 * ibis.interval(days=1), order_by=time_table.time
    )
    expr = time_table['value'].mean().over(window)
    # result should adjust time context accordingly
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result, expected)
