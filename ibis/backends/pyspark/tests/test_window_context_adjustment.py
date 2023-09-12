from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis

pyspark = pytest.importorskip("pyspark")

import pyspark.sql.functions as F  # noqa: E402
from pyspark.sql.window import Window  # noqa: E402


@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [
        ([(ibis.interval(hours=1), 0)], (-3600, 0)),  # 1h back looking window
        ([(ibis.interval(hours=2), 0)], (-7200, 0)),  # 2h back looking window
        (
            [(0, ibis.interval(hours=1))],
            (0, 3600),
        ),  # 1h forward looking window
        (
            [(ibis.interval(hours=1), ibis.interval(hours=1))],
            (-3600, 3600),
        ),  # both forward and trailing
    ],
    indirect=["ibis_windows"],
)
def test_window_with_timecontext(con, ibis_windows, spark_range):
    """Test context adjustment for trailing / range window.

    We expand context according to window sizes, for example, for a table of:
    time       value
    2020-01-01   a
    2020-01-02   b
    2020-01-03   c
    2020-01-04   d
    with context = (2020-01-03, 2002-01-04) trailing count for 1 day will be:
    time       value  count
    2020-01-03   c      2
    2020-01-04   d      2
    trailing count for 2 days will be:
    time       value  count
    2020-01-03   c      3
    2020-01-04   d      3
    with context = (2020-01-01, 2002-01-02) count for 1 day forward looking
    window will be:
    time       value  count
    2020-01-01   a      2
    2020-01-02   b      2
    """
    table = con.table("time_indexed_table")
    context = (
        pd.Timestamp("20170102 07:00:00", tz="UTC"),
        pd.Timestamp("20170103", tz="UTC"),
    )
    result_pd = table.mutate(
        count=table["value"].count().over(ibis_windows[0])
    ).execute(timecontext=context)
    spark_table = table.compile()
    spark_window = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range)
    )
    expected = spark_table.withColumn(
        "count",
        F.count(spark_table["value"]).over(spark_window),
    ).toPandas()
    expected = expected[
        expected.time.between(*(t.tz_convert(None) for t in context))
    ].reset_index(drop=True)
    tm.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [([(None, 0)], (Window.unboundedPreceding, 0))],
    indirect=["ibis_windows"],
)
def test_cumulative_window(con, ibis_windows, spark_range):
    """Test context adjustment for cumulative window.

    For cumulative window, by definition we should look back infinitely.
    When data is trimmed by time context, we define the limit of looking
    back is the start time of given time context. Thus for a table of
    time       value
    2020-01-01   a
    2020-01-02   b
    2020-01-03   c
    2020-01-04   d
    with context = (2020-01-02, 2002-01-03) cumulative count will be:
    time       value  count
    2020-01-02   b      1
    2020-01-03   c      2
    """
    table = con.table("time_indexed_table")
    context = (
        pd.Timestamp("20170102 07:00:00", tz="UTC"),
        pd.Timestamp("20170105", tz="UTC"),
    )
    result_pd = table.mutate(
        count_cum=table["value"].count().over(ibis_windows[0])
    ).execute(timecontext=context)

    spark_table = table.compile(timecontext=context)
    spark_window = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range)
    )
    expected = spark_table.withColumn(
        "count_cum",
        F.count(spark_table["value"]).over(spark_window),
    ).toPandas()
    expected = expected[
        expected.time.between(*(t.tz_convert(None) for t in context))
    ].reset_index(drop=True)
    tm.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [
        (
            [(ibis.interval(hours=1), 0), (ibis.interval(hours=2), 0)],
            [(-3600, 0), (-7200, 0)],
        )
    ],
    indirect=["ibis_windows"],
)
def test_multiple_trailing_window(con, ibis_windows, spark_range):
    """Test context adjustment for multiple trailing window.

    When there are multiple window ops, we need to verify contexts are
    adjusted correctly for all windows. In this tests we are constructing
    one trailing window for 1h and another trailing window for 2h
    """
    table = con.table("time_indexed_table")
    context = (
        pd.Timestamp("20170102 07:00:00", tz="UTC"),
        pd.Timestamp("20170105", tz="UTC"),
    )
    result_pd = table.mutate(
        count_1h=table["value"].count().over(ibis_windows[0]),
        count_2h=table["value"].count().over(ibis_windows[1]),
    ).execute(timecontext=context)

    spark_table = table.compile()
    spark_window_1h = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[0])
    )
    spark_window_2h = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[1])
    )
    expected = (
        spark_table.withColumn(
            "count_1h", F.count(spark_table["value"]).over(spark_window_1h)
        )
        .withColumn("count_2h", F.count(spark_table["value"]).over(spark_window_2h))
        .toPandas()
    )
    expected = expected[
        expected.time.between(*(t.tz_convert(None) for t in context))
    ].reset_index(drop=True)
    tm.assert_frame_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [
        (
            [(ibis.interval(hours=1), 0), (ibis.interval(hours=2), 0)],
            [(-3600, 0), (-7200, 0)],
        )
    ],
    indirect=["ibis_windows"],
)
def test_chained_trailing_window(con, ibis_windows, spark_range):
    """Test context adjustment for chained windows.

    When there are chained window ops, we need to verify contexts are
    adjusted correctly for all windows. In this tests we are constructing
    one trailing window for 1h and trailing window on the new column for
    2h
    """
    table = con.table("time_indexed_table")
    context = (
        pd.Timestamp("20170102 07:00:00", tz="UTC"),
        pd.Timestamp("20170105", tz="UTC"),
    )
    table = table.mutate(
        new_col=table["value"].count().over(ibis_windows[0]),
    )
    table = table.mutate(count=table["new_col"].count().over(ibis_windows[1]))
    result_pd = table.execute(timecontext=context)

    spark_table = table.compile()
    spark_window_1h = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[0])
    )
    spark_window_2h = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[1])
    )
    spark_table = spark_table.withColumn(
        "new_col", F.count(spark_table["value"]).over(spark_window_1h)
    )
    spark_table = spark_table.withColumn(
        "count", F.count(spark_table["new_col"]).over(spark_window_2h)
    )
    expected = spark_table.toPandas()
    expected = expected[
        expected.time.between(*(t.tz_convert(None) for t in context))
    ].reset_index(drop=True)
    tm.assert_frame_equal(result_pd, expected)


@pytest.mark.xfail(
    reason="Issue #2457 Adjust context properly for mixed rolling window,"
    " cumulative window and non window ops",
    strict=True,
)
@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [
        (
            [(ibis.interval(hours=1), 0), (None, 0)],
            [(-3600, 0), (Window.unboundedPreceding, 0)],
        )
    ],
    indirect=["ibis_windows"],
)
def test_rolling_with_cumulative_window(con, ibis_windows, spark_range):
    """Test context adjustment for rolling window and cumulative window.

    cumulative window should calculate only with in user's context,
    while rolling window should calculate on expanded context.
    For a rolling window of 1 day,
    time       value
    2020-01-01   a
    2020-01-02   b
    2020-01-03   c
    2020-01-04   d
    with context = (2020-01-02, 2002-01-03), count will be:
    time       value  roll_count cum_count
    2020-01-02   b      2            1
    2020-01-03   c      2            2
    """
    table = con.table("time_indexed_table")
    context = (
        pd.Timestamp("20170102 07:00:00", tz="UTC"),
        pd.Timestamp("20170105", tz="UTC"),
    )
    result_pd = table.mutate(
        count_1h=table["value"].count().over(ibis_windows[0]),
        count_cum=table["value"].count().over(ibis_windows[1]),
    ).execute(timecontext=context)

    spark_table = table.compile()
    spark_window_1h = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[0])
    )
    spark_window_cum = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[1])
    )
    expected = (
        spark_table.withColumn(
            "count_1h", F.count(spark_table["value"]).over(spark_window_1h)
        )
        .withColumn("count_cum", F.count(spark_table["value"]).over(spark_window_cum))
        .toPandas()
    )
    expected = expected[
        expected.time.between(*(t.tz_convert(None) for t in context))
    ].reset_index(drop=True)
    tm.assert_frame_equal(result_pd, expected)


@pytest.mark.xfail(
    reason="Issue #2457 Adjust context properly for mixed rolling window,"
    " cumulative window and non window ops",
    strict=True,
)
@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [([(ibis.interval(hours=1), 0)], [(-3600, 0)])],
    indirect=["ibis_windows"],
)
def test_rolling_with_non_window_op(con, ibis_windows, spark_range):
    """Test context adjustment for rolling window and non window ops.

    non window ops should calculate only with in user's context,
    while rolling window should calculate on expanded context.
    For a rolling window of 1 day, and a `count` aggregation
    time       value
    2020-01-01   a
    2020-01-02   b
    2020-01-03   c
    2020-01-04   d
    with context = (2020-01-02, 2002-01-04), result will be:
    time       value  roll_count    count
    2020-01-02   b      2            3
    2020-01-03   c      2            3
    2020-01-04   d      2            3
    Because there are 3 rows within user context (01-02, 01-04),
    count should return 3 for every row, rather 4, based on the
    adjusted context (01-01, 01-04).
    """
    table = con.table("time_indexed_table")
    context = (
        pd.Timestamp("20170102 07:00:00", tz="UTC"),
        pd.Timestamp("20170105", tz="UTC"),
    )
    result_pd = table.mutate(
        count_1h=table["value"].count().over(ibis_windows[0]),
        count=table["value"].count(),
    ).execute(timecontext=context)

    spark_table = table.compile()
    spark_window_1h = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[0])
    )
    expected = (
        spark_table.withColumn(
            "count_1h", F.count(spark_table["value"]).over(spark_window_1h)
        )
        .withColumn("count", F.count(spark_table["value"]))
        .toPandas()
    )
    expected = expected[
        expected.time.between(*(t.tz_convert(None) for t in context))
    ].reset_index(drop=True)
    tm.assert_frame_equal(result_pd, expected)


def test_complex_window(con):
    """Test window with different sizes mix context adjustment for window op
    that require context adjustment and non window op that doesn't adjust
    context."""
    table = con.table("time_indexed_table")
    context = (
        pd.Timestamp("20170102 07:00:00", tz="UTC"),
        pd.Timestamp("20170105", tz="UTC"),
    )
    window = ibis.trailing_window(
        preceding=ibis.interval(hours=1), order_by="time", group_by="key"
    )
    window2 = ibis.trailing_window(
        preceding=ibis.interval(hours=2), order_by="time", group_by="key"
    )
    window_cum = ibis.cumulative_window(order_by="time", group_by="key")
    # context should be adjusted accordingly for each window
    result_pd = (
        table.mutate(
            count_1h=table["value"].count().over(window),
            count_2h=table["value"].count().over(window2),
            count_cum=table["value"].count().over(window_cum),
        )
        .mutate(count=table["value"].count())
        .execute(timecontext=context)
    )

    df = table.execute()
    expected_win_1h = (
        df.set_index("time")
        .groupby("key")
        .value.rolling("1h", closed="both")
        .count()
        .rename("count_1h")
        .astype(int)
    )
    expected_win_2h = (
        df.set_index("time")
        .groupby("key")
        .value.rolling("2h", closed="both")
        .count()
        .rename("count_2h")
        .astype(int)
    )
    expected_cum_win = (
        df.set_index("time")
        .groupby("key")
        .value.expanding()
        .count()
        .rename("count_cum")
        .astype(int)
    )
    df = df.set_index("time")
    df = df.assign(
        count_1h=expected_win_1h.sort_index(level=["time", "key"]).reset_index(
            level="key", drop=True
        )
    )
    df = df.assign(
        count_2h=expected_win_2h.sort_index(level=["time", "key"]).reset_index(
            level="key", drop=True
        )
    )
    df = df.assign(
        count_cum=expected_cum_win.sort_index(level=["time", "key"]).reset_index(
            level="key", drop=True
        )
    )
    df["count"] = df.groupby(["key"])["value"].transform("count")
    df = df.reset_index()
    expected = (
        df[df.time.between(*(t.tz_convert(None) for t in context))]
        .sort_values(["key"])
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result_pd, expected)
