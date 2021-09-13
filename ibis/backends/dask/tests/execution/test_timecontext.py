import dask.dataframe as dd
import pytest
from dask.dataframe.utils import tm
from pandas import Timedelta, Timestamp

import ibis
import ibis.common.exceptions as com
from ibis.expr.timecontext import TimeContextRelation, compare_timecontext


def test_execute_with_timecontext(time_table):
    expr = time_table
    # define a time context for time-series data
    context = (Timestamp('20170101'), Timestamp('20170103'))

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
        context = Timestamp('20101010')
        expr.execute(timecontext=context)

    # define context with begin value later than end
    with pytest.raises(com.IbisError, match=r".*before or equal.*"):
        context = (Timestamp('20101010'), Timestamp('20090101'))
        expr.execute(timecontext=context)

    # execute context with a table without time column
    with pytest.raises(com.IbisError, match=r".*must have a time column.*"):
        context = (Timestamp('20090101'), Timestamp('20100101'))
        t.execute(timecontext=context)


def test_compare_timecontext():
    c1 = (Timestamp('20170101'), Timestamp('20170103'))
    c2 = (Timestamp('20170101'), Timestamp('20170111'))
    c3 = (Timestamp('20160101'), Timestamp('20160103'))
    c4 = (Timestamp('20161215'), Timestamp('20170102'))
    assert compare_timecontext(c1, c2) == TimeContextRelation.SUBSET
    assert compare_timecontext(c2, c1) == TimeContextRelation.SUPERSET
    assert compare_timecontext(c1, c4) == TimeContextRelation.OVERLAP
    assert compare_timecontext(c1, c3) == TimeContextRelation.NONOVERLAP


def test_context_adjustment_asof_join(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(
        time_keyed_right, 'time', by='key', tolerance=4 * ibis.interval(days=1)
    )[time_keyed_left, time_keyed_right.other_value]
    context = (Timestamp('20170105'), Timestamp('20170111'))
    result = expr.execute(timecontext=context)

    # compare with asof_join of manually trimmed tables
    trimmed_df1 = time_keyed_df1[time_keyed_df1['time'] >= context[0]][
        time_keyed_df1['time'] < context[1]
    ]
    trimmed_df2 = time_keyed_df2[
        time_keyed_df2['time'] >= context[0] - Timedelta(days=4)
    ][time_keyed_df2['time'] < context[1]]
    expected = dd.merge_asof(
        trimmed_df1,
        trimmed_df2,
        on='time',
        by='key',
        tolerance=Timedelta('4D'),
    ).compute()
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="TODO - windowing - #2553")
@pytest.mark.parametrize(
    ['interval_ibis', 'interval_pd'],
    [
        (ibis.interval(days=1), '1d'),
        (3 * ibis.interval(days=1), '3d'),
        (5 * ibis.interval(days=1), '5d'),
    ],
)
def test_context_adjustment_window(
    time_table, time_df3, interval_ibis, interval_pd
):
    # trim data manually
    expected = (
        time_df3.set_index('time')
        .value.rolling(interval_pd, closed='both')
        .mean()
    )
    expected = expected[expected.index >= Timestamp('20170105')].reset_index(
        drop=True
    )

    context = Timestamp('20170105'), Timestamp('20170111')

    window = ibis.trailing_window(interval_ibis, order_by=time_table.time)
    expr = time_table['value'].mean().over(window)
    # result should adjust time context accordingly
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(reason="TODO - windowing - #2553")
def test_setting_timecontext_in_scope(time_table, time_df3):
    expected_win_1 = (
        time_df3.compute()
        .set_index('time')
        .value.rolling('3d', closed='both')
        .mean()
    )
    expected_win_1 = expected_win_1[
        expected_win_1.index >= Timestamp('20170105')
    ].reset_index(drop=True)

    context = Timestamp('20170105'), Timestamp('20170111')
    window1 = ibis.trailing_window(
        3 * ibis.interval(days=1), order_by=time_table.time
    )
    """
    In the following expression, Selection node will be executed first and
    get table in context ('20170105', '20170101'). Then in window execution
    table will be executed again with a larger context adjusted by window
    preceeding days ('20170102', '20170111'). To get the correct result,
    the cached table result with a smaller context must be discard and updated
    to a larger time range.
    """
    expr = time_table.mutate(value=time_table['value'].mean().over(window1))
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result["value"], expected_win_1)


@pytest.mark.xfail(reason="TODO - windowing - #2553")
def test_context_adjustment_multi_window(time_table, time_df3):
    expected_win_1 = (
        time_df3.compute()
        .set_index('time')
        .rename(columns={'value': 'v1'})['v1']
        .rolling('3d', closed='both')
        .mean()
    )
    expected_win_1 = expected_win_1[
        expected_win_1.index >= Timestamp('20170105')
    ].reset_index(drop=True)

    expected_win_2 = (
        time_df3.compute()
        .set_index('time')
        .rename(columns={'value': 'v2'})['v2']
        .rolling('2d', closed='both')
        .mean()
    )
    expected_win_2 = expected_win_2[
        expected_win_2.index >= Timestamp('20170105')
    ].reset_index(drop=True)

    context = Timestamp('20170105'), Timestamp('20170111')
    window1 = ibis.trailing_window(
        3 * ibis.interval(days=1), order_by=time_table.time
    )
    window2 = ibis.trailing_window(
        2 * ibis.interval(days=1), order_by=time_table.time
    )
    expr = time_table.mutate(
        v1=time_table['value'].mean().over(window1),
        v2=time_table['value'].mean().over(window2),
    )
    result = expr.execute(timecontext=context)

    tm.assert_series_equal(result["v1"], expected_win_1)
    tm.assert_series_equal(result["v2"], expected_win_2)


@pytest.mark.xfail(reason="TODO - windowing - #2553")
def test_context_adjustment_window_groupby_id(time_table, time_df3):
    """This test case is meant to test trim_window_result method
    in dask/execution/window.py to see if it could trim Series
    correctly with groupby params
    """
    expected = (
        time_df3.compute()
        .set_index('time')
        .groupby('id')
        .value.rolling('3d', closed='both')
        .mean()
    )
    # This is a MultiIndexed Series
    expected = expected.reset_index()
    expected = expected[expected.time >= Timestamp('20170105')].reset_index(
        drop=True
    )['value']

    context = Timestamp('20170105'), Timestamp('20170111')

    # expected.index.name = None
    window = ibis.trailing_window(
        3 * ibis.interval(days=1), group_by='id', order_by=time_table.time
    )
    expr = time_table['value'].mean().over(window)
    # result should adjust time context accordingly
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result, expected)
