from __future__ import annotations

import pandas as pd
import pytest
from packaging.version import parse as vparse

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.df.scope import Scope
from ibis.backends.base.df.timecontext import (
    TimeContext,
    TimeContextRelation,
    adjust_context,
    compare_timecontext,
    construct_time_context_aware_series,
)
from ibis.backends.pandas.execution import execute
from ibis.backends.pandas.execution.window import trim_window_result
from ibis.backends.pandas.tests.conftest import TestConf as tm


class CustomAsOfJoin(ops.AsOfJoin):
    pass


def test_execute_with_timecontext(time_table):
    expr = time_table
    # define a time context for time-series data
    context = (pd.Timestamp("20170101"), pd.Timestamp("20170103"))

    # without time context, execute produces every row
    df_all = expr.execute()
    assert len(df_all["time"]) == 8

    # with context set, execute produces only rows within context
    df_within_context = expr.execute(timecontext=context)
    assert len(df_within_context["time"]) == 1


def test_bad_timecontext(time_table, t):
    expr = time_table

    # define context with illegal string
    with pytest.raises(com.IbisError, match=r".*type pd.Timestamp.*"):
        context = ("bad", "context")
        expr.execute(timecontext=context)

    # define context with unsupported type int
    with pytest.raises(com.IbisError, match=r".*type pd.Timestamp.*"):
        context = (20091010, 20100101)
        expr.execute(timecontext=context)

    # define context with too few values
    with pytest.raises(com.IbisError, match=r".*should specify.*"):
        context = pd.Timestamp("20101010")
        expr.execute(timecontext=context)

    # define context with begin value later than end
    with pytest.raises(com.IbisError, match=r".*before or equal.*"):
        context = (pd.Timestamp("20101010"), pd.Timestamp("20090101"))
        expr.execute(timecontext=context)

    # execute context with a table without TIME_COL
    with pytest.raises(com.IbisError, match=r".*must have a time column.*"):
        context = (pd.Timestamp("20090101"), pd.Timestamp("20100101"))
        t.execute(timecontext=context)


def test_bad_call_to_adjust_context():
    op = "not_a_node"
    context = (pd.Timestamp("20170101"), pd.Timestamp("20170103"))
    scope = Scope()
    with pytest.raises(
        com.IbisError, match=r".*Unsupported input type for adjust context.*"
    ):
        adjust_context(op, scope, context)


def test_compare_timecontext():
    c1 = (pd.Timestamp("20170101"), pd.Timestamp("20170103"))
    c2 = (pd.Timestamp("20170101"), pd.Timestamp("20170111"))
    c3 = (pd.Timestamp("20160101"), pd.Timestamp("20160103"))
    c4 = (pd.Timestamp("20161215"), pd.Timestamp("20170102"))
    assert compare_timecontext(c1, c2) == TimeContextRelation.SUBSET
    assert compare_timecontext(c2, c1) == TimeContextRelation.SUPERSET
    assert compare_timecontext(c1, c4) == TimeContextRelation.OVERLAP
    assert compare_timecontext(c1, c3) == TimeContextRelation.NONOVERLAP


def test_context_adjustment_asof_join(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(
        time_keyed_right, "time", by="key", tolerance=4 * ibis.interval(days=1)
    )[time_keyed_left, time_keyed_right.other_value]
    context = (pd.Timestamp("20170105"), pd.Timestamp("20170111"))
    result = expr.execute(timecontext=context)

    # compare with asof_join of manually trimmed tables
    trimmed_df1 = time_keyed_df1[time_keyed_df1["time"] >= context[0]][
        time_keyed_df1["time"] < context[1]
    ]
    trimmed_df2 = time_keyed_df2[
        time_keyed_df2["time"] >= context[0] - pd.Timedelta(days=4)
    ][time_keyed_df2["time"] < context[1]]
    expected = pd.merge_asof(
        trimmed_df1,
        trimmed_df2,
        on="time",
        by="key",
        tolerance=pd.Timedelta("4D"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ["interval_ibis", "interval_pd"],
    [
        (ibis.interval(days=1), "1d"),
        (3 * ibis.interval(days=1), "3d"),
        (5 * ibis.interval(days=1), "5d"),
    ],
)
def test_context_adjustment_window(time_table, time_df3, interval_ibis, interval_pd):
    # trim data manually
    expected = (
        time_df3.set_index("time").value.rolling(interval_pd, closed="both").mean()
    )
    expected = expected[expected.index >= pd.Timestamp("20170105")].reset_index(
        drop=True
    )

    context = pd.Timestamp("20170105"), pd.Timestamp("20170111")

    window = ibis.trailing_window(interval_ibis, order_by=time_table.time)
    expr = time_table["value"].mean().over(window)
    # result should adjust time context accordingly
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result, expected)


def test_trim_window_result(time_df3):
    """Unit test `trim_window_result` in Window execution."""
    df = time_df3.copy()
    context = pd.Timestamp("20170105"), pd.Timestamp("20170111")

    # trim_window_result takes a MultiIndex Series as input
    series = df["value"]
    time_index = df.set_index("time").index
    series.index = pd.MultiIndex.from_arrays(
        [series.index, time_index],
        names=series.index.names + ["time"],
    )
    result = trim_window_result(series, context)
    expected = df["time"][df["time"] >= pd.Timestamp("20170105")].reset_index(drop=True)

    # result should adjust time context accordingly
    tm.assert_series_equal(result.reset_index()["time"], expected)

    # trim with a non-datetime type of 'time' throws Exception
    wrong_series = df["id"]
    df["time"] = df["time"].astype(str)
    time_index = df.set_index("time").index
    wrong_series.index = pd.MultiIndex.from_arrays(
        [wrong_series.index, time_index],
        names=wrong_series.index.names + ["time"],
    )
    with pytest.raises(TypeError, match=r".*not supported between instances.*"):
        trim_window_result(wrong_series, context)

    # column is ignored and series is not trimmed
    no_context_result = trim_window_result(series, None)
    tm.assert_series_equal(no_context_result, series)


def test_setting_timecontext_in_scope(time_table, time_df3):
    expected_win_1 = (
        time_df3.set_index("time").value.rolling("3d", closed="both").mean()
    )
    expected_win_1 = expected_win_1[
        expected_win_1.index >= pd.Timestamp("20170105")
    ].reset_index(drop=True)

    context = pd.Timestamp("20170105"), pd.Timestamp("20170111")
    window1 = ibis.trailing_window(3 * ibis.interval(days=1), order_by=time_table.time)
    """In the following expression, Selection node will be executed first and
    get table in context ('20170105', '20170101').

    Then in window execution table will be executed again with a larger
    context adjusted by window preceding days ('20170102', '20170111').
    To get the correct result, the cached table result with a smaller
    context must be discard and updated to a larger time range.
    """
    expr = time_table.mutate(value=time_table["value"].mean().over(window1))
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result["value"], expected_win_1)


def test_context_adjustment_multi_window(time_table, time_df3):
    expected_win_1 = (
        time_df3.set_index("time")
        .rename(columns={"value": "v1"})["v1"]
        .rolling("3d", closed="both")
        .mean()
    )
    expected_win_1 = expected_win_1[
        expected_win_1.index >= pd.Timestamp("20170105")
    ].reset_index(drop=True)

    expected_win_2 = (
        time_df3.set_index("time")
        .rename(columns={"value": "v2"})["v2"]
        .rolling("2d", closed="both")
        .mean()
    )
    expected_win_2 = expected_win_2[
        expected_win_2.index >= pd.Timestamp("20170105")
    ].reset_index(drop=True)

    context = pd.Timestamp("20170105"), pd.Timestamp("20170111")
    window1 = ibis.trailing_window(3 * ibis.interval(days=1), order_by=time_table.time)
    window2 = ibis.trailing_window(2 * ibis.interval(days=1), order_by=time_table.time)
    expr = time_table.mutate(
        v1=time_table["value"].mean().over(window1),
        v2=time_table["value"].mean().over(window2),
    )
    result = expr.execute(timecontext=context)

    tm.assert_series_equal(result["v1"], expected_win_1)
    tm.assert_series_equal(result["v2"], expected_win_2)


@pytest.mark.xfail(
    condition=vparse("1.4") <= vparse(pd.__version__) < vparse("1.4.2"),
    raises=ValueError,
    reason="https://github.com/pandas-dev/pandas/pull/44068",
)
def test_context_adjustment_window_groupby_id(time_table, time_df3):
    """This test case is meant to test trim_window_result method in
    pandas/execution/window.py to see if it could trim Series correctly with
    groupby params."""
    expected = (
        time_df3.set_index("time")
        .groupby("id")
        .value.rolling("3d", closed="both")
        .mean()
    )
    # This is a MultiIndexed Series
    expected = expected.reset_index()
    expected = expected[expected.time >= pd.Timestamp("20170105")].reset_index(
        drop=True
    )["value"]

    context = pd.Timestamp("20170105"), pd.Timestamp("20170111")

    # expected.index.name = None
    window = ibis.trailing_window(
        3 * ibis.interval(days=1), group_by="id", order_by=time_table.time
    )
    expr = time_table["value"].mean().over(window)
    # result should adjust time context accordingly
    result = expr.execute(timecontext=context)
    tm.assert_series_equal(result, expected)


def test_adjust_context_scope(time_keyed_left, time_keyed_right):
    """Test that `adjust_context` has access to `scope` by default."""

    @adjust_context.register(CustomAsOfJoin)
    def adjust_context_custom_asof_join(
        op: ops.AsOfJoin,
        scope: Scope,
        timecontext: TimeContext,
    ) -> TimeContext:
        """Confirms that `scope` is passed in."""
        assert scope is not None
        return timecontext

    expr = CustomAsOfJoin(
        left=time_keyed_left,
        right=time_keyed_right,
        predicates="time",
        by="key",
        tolerance=ibis.interval(days=4),
    ).to_expr()
    expr = expr[time_keyed_left, time_keyed_right.other_value]
    context = (pd.Timestamp("20170105"), pd.Timestamp("20170111"))
    expr.execute(timecontext=context)


def test_adjust_context_complete_shift(
    time_keyed_left,
    time_keyed_right,
    time_keyed_df1,
    time_keyed_df2,
):
    """Test `adjust_context` function that completely shifts the context.

    This results in an adjusted context that is NOT a subset of the
    original context. This is unlike an `adjust_context` function
    that only expands the context.

    See #3104
    """

    # Create a contrived `adjust_context` function for
    # CustomAsOfJoin to mock this.

    @adjust_context.register(CustomAsOfJoin)
    def adjust_context_custom_asof_join(
        op: ops.AsOfJoin,
        scope: Scope,
        timecontext: TimeContext,
    ) -> TimeContext:
        """Shifts both the begin and end in the same direction."""

        begin, end = timecontext
        timedelta = execute(op.tolerance)
        return (begin - timedelta, end - timedelta)

    expr = CustomAsOfJoin(
        left=time_keyed_left,
        right=time_keyed_right,
        predicates="time",
        by="key",
        tolerance=ibis.interval(days=4),
    ).to_expr()
    expr = expr[time_keyed_left, time_keyed_right.other_value]
    context = (pd.Timestamp("20170101"), pd.Timestamp("20170111"))
    result = expr.execute(timecontext=context)

    # Compare with asof_join of manually trimmed tables
    # Left table: No shift for context
    # Right table: Shift both begin and end of context by 4 days
    trimmed_df1 = time_keyed_df1[time_keyed_df1["time"] >= context[0]][
        time_keyed_df1["time"] < context[1]
    ]
    trimmed_df2 = time_keyed_df2[
        time_keyed_df2["time"] >= context[0] - pd.Timedelta(days=4)
    ][time_keyed_df2["time"] < context[1] - pd.Timedelta(days=4)]
    expected = pd.merge_asof(
        trimmed_df1,
        trimmed_df2,
        on="time",
        by="key",
        tolerance=pd.Timedelta("4D"),
    )

    tm.assert_frame_equal(result, expected)


def test_construct_time_context_aware_series(time_df3):
    """Unit test for `construct_time_context_aware_series`"""
    # Series without 'time' index will result in a MultiIndex with 'time'
    df = time_df3
    expected = df["value"]
    time_index = pd.Index(df["time"])
    expected.index = pd.MultiIndex.from_arrays(
        [expected.index, time_index],
        names=expected.index.names + ["time"],
    )
    result = construct_time_context_aware_series(df["value"], df)
    tm.assert_series_equal(result, expected)

    # Series with 'time' as index will not change
    time_indexed_df = time_df3.set_index("time")
    expected_time_aware = time_indexed_df["value"]
    result_time_aware = construct_time_context_aware_series(
        time_indexed_df["value"], time_indexed_df
    )
    tm.assert_series_equal(result_time_aware, expected_time_aware)

    # Series with a MultiIndex, where 'time' is in the MultiIndex,
    # will not change
    multi_index_time_aware_series = result_time_aware
    expected_multi_index_time_aware = result_time_aware
    result_multi_index_time_aware = construct_time_context_aware_series(
        multi_index_time_aware_series, time_indexed_df
    )
    tm.assert_series_equal(
        result_multi_index_time_aware, expected_multi_index_time_aware
    )

    # Series with a MultiIndex, where 'time' is NOT in the MultiIndex,
    # 'time' will be added into the MultiIndex
    multi_index_series = df["id"]
    expected_multi_index = df["id"].copy()
    other_index = pd.Index(df["value"])
    expected_multi_index.index = pd.MultiIndex.from_arrays(
        [expected_multi_index.index, other_index, time_index],
        names=expected_multi_index.index.names + ["value", "time"],
    )
    multi_index_series.index = pd.MultiIndex.from_arrays(
        [multi_index_series.index, other_index],
        names=multi_index_series.index.names + ["value"],
    )
    result_multi_index = construct_time_context_aware_series(multi_index_series, df)
    tm.assert_series_equal(result_multi_index, expected_multi_index)
