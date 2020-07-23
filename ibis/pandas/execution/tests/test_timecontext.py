import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis
import ibis.common.exceptions as com

pytestmark = pytest.mark.pandas


def test_execute_with_timecontext(time_table):
    expr = time_table
    # define a time context for time-series data
    context = (pd.Timestamp('20170101'), pd.Timestamp('20170103'))

    # without time context, execute produces every row
    df_all = expr.execute()
    assert len(df_all['time']) == 8

    # with context set, execute produces only rows within context
    df_within_context = expr.execute(timecontext=context)
    assert len(df_within_context['time']) == 1


def test_bad_timecontext(time_table, t):
    expr = time_table

    # define context with illegal string
    with pytest.raises(com.IbisError, match=r".*type pd.Timestamp.*"):
        context = ('bad', 'context')
        expr.execute(timecontext=context)

    # define context with unsupport type int
    with pytest.raises(com.IbisError, match=r".*type pd.Timestamp.*"):
        context = (20091010, 20100101)
        expr.execute(timecontext=context)

    # define context with too few values
    with pytest.raises(com.IbisError, match=r".*should specify.*"):
        context = pd.Timestamp('20101010')
        expr.execute(timecontext=context)

    # define context with begin value later than end
    with pytest.raises(com.IbisError, match=r".*before or equal.*"):
        context = (pd.Timestamp('20101010'), pd.Timestamp('20090101'))
        expr.execute(timecontext=context)

    # execute context with a table without TIME_COL
    with pytest.raises(com.IbisError, match=r".*must have a time column.*"):
        context = (pd.Timestamp('20090101'), pd.Timestamp('20100101'))
        t.execute(timecontext=context)


def test_context_adjustment_asof_join(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(
        time_keyed_right, 'time', by='key', tolerance=4 * ibis.interval(days=1)
    )[time_keyed_left, time_keyed_right.other_value]
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
