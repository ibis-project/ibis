import pandas as pd
import pandas.util.testing as tm
import pyspark.sql.functions as F
import pytest
from pyspark.sql import Window
from pytest import param

import ibis

pytest.importorskip('pyspark')
pytestmark = pytest.mark.pyspark


def test_table_with_timecontext(client):
    table = client.table('time_indexed_table')
    context = (pd.Timestamp('20170102'), pd.Timestamp('20170103'))
    result = table.compile(timecontext=context).toPandas()
    expected = table.compile().toPandas()
    expected = expected[expected.time.between(*context)]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('ibis_window', 'spark_range'),
    [
        param(
            ibis.trailing_window(
                preceding=ibis.interval(hours=1),
                order_by='time',
                group_by='key',
            ),
            (-3600, 0),
        ),  # 1h back looking window
        param(
            ibis.trailing_window(
                preceding=ibis.interval(hours=2),
                order_by='time',
                group_by='key',
            ),
            (-7200, 0),
        ),  # 2h back looking window
        param(
            ibis.range_window(
                preceding=0,
                following=ibis.interval(hours=1),
                order_by='time',
                group_by='key',
            ),
            (0, 3600),
        ),  # 1h forward looking window
    ],
)
def test_window_with_timecontext(client, ibis_window, spark_range):
    """ Test context adjustment for trailing / range window

    We expand context according to window sizes, for example, for a table of:
    time       value
    2020-01-01   a
    2020-01-02   b
    2020-01-03   c
    2020-01-04   d
    with context = (2020-01-03, 2002-01-04) trailing count for 1 day will be:
    time       value  count
    2020-01-03   c      2
    2020-01-04   d      3
    trailing count for 2 days will be:
    time       value  count
    2020-01-03   c      3
    2020-01-04   d      4
    with context = (2020-01-01, 2002-01-02) count for 1 day forward looking
    window will be:
    time       value  count
    2020-01-01   a      2
    2020-01-02   b      2
    """
    table = client.table('time_indexed_table')
    context = (
        pd.Timestamp('20170102 07:00:00', tz='UTC'),
        pd.Timestamp('20170103', tz='UTC'),
    )
    result = table.mutate(
        count=table['value'].count().over(ibis_window)
    ).compile(timecontext=context)
    result_pd = result.toPandas()
    spark_table = table.compile()
    spark_window = (
        Window.partitionBy('key')
        .orderBy(F.col('time').cast('long'))
        .rangeBetween(*spark_range)
    )
    expected = spark_table.withColumn(
        'count', F.count(spark_table['value']).over(spark_window),
    ).toPandas()
    expected = expected[
        expected.time.between(*(t.tz_convert(None) for t in context))
    ].reset_index(drop=True)
    tm.assert_frame_equal(result_pd, expected)


def test_cumulative_window(client):
    """ Test context adjustment for cumulative window

    For cumulative window, by defination we should look back infinately.
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
    table = client.table('time_indexed_table')
    context = (
        pd.Timestamp('20170102 07:00:00', tz='UTC'),
        pd.Timestamp('20170105', tz='UTC'),
    )
    window = ibis.cumulative_window(order_by='time', group_by='key')
    result = table.mutate(
        mean_cum=table['value'].count().over(window)
    ).compile(timecontext=context)
    result_pd = result.toPandas()
    df = table.compile().toPandas()
    df = df[df.time.between(*(t.tz_convert(None) for t in context))]
    expected_cum_win = (
        df.set_index('time')
        .groupby('key')
        .value.expanding()
        .count()
        .rename('count_cum')
        .astype(int)
    )
    df = df.set_index('time')
    df = df.assign(
        mean_cum=expected_cum_win.sort_index(
            level=['time', 'key']
        ).reset_index(level='key', drop=True)
    )
    expected = df.sort_values(by=['key', 'time']).reset_index()
    tm.assert_frame_equal(result_pd, expected)


def test_complex_window(client):
    table = client.table('time_indexed_table')
    context = (
        pd.Timestamp('20170102 07:00:00', tz='UTC'),
        pd.Timestamp('20170105', tz='UTC'),
    )
    window = ibis.trailing_window(
        preceding=ibis.interval(hours=1), order_by='time', group_by='key'
    )
    window2 = ibis.trailing_window(
        preceding=ibis.interval(hours=2), order_by='time', group_by='key'
    )
    window_cum = ibis.cumulative_window(order_by='time', group_by='key')
    # context should be adjusted accordingly for each window
    result = (
        table.mutate(count_1h=table['value'].count().over(window))
        .mutate(
            count_2h=table['value'].count().over(window2),
            count_cum=table['value'].count().over(window_cum),
        )
        .mutate(mean_1h=table['value'].mean().over(window))
        .compile(timecontext=context)
    )
    result_pd = result.toPandas()

    df = table.compile().toPandas()
    expected_win_1h = (
        df.set_index('time')
        .groupby('key')
        .value.rolling('1h', closed='both')
        .count()
        .rename('count_1h')
        .astype(int)
    )
    expected_win_2h = (
        df.set_index('time')
        .groupby('key')
        .value.rolling('2h', closed='both')
        .count()
        .rename('count_2h')
        .astype(int)
    )
    expected_win_mean_1h = (
        df.set_index('time')
        .groupby('key')
        .value.rolling('1h', closed='both')
        .mean()
        .rename('mean_1h')
    )
    expected_cum_win = (
        df.set_index('time')
        .groupby('key')
        .value.expanding()
        .count()
        .rename('count_cum')
        .astype(int)
    )
    df = df.set_index('time')
    df = df.assign(
        count_1h=expected_win_1h.sort_index(level=['time', 'key']).reset_index(
            level='key', drop=True
        )
    )
    df = df.assign(
        count_2h=expected_win_2h.sort_index(level=['time', 'key']).reset_index(
            level='key', drop=True
        )
    )
    df = df.assign(
        count_cum=expected_cum_win.sort_index(
            level=['time', 'key']
        ).reset_index(level='key', drop=True)
    )
    df = df.assign(
        mean_1h=expected_win_mean_1h.sort_index(
            level=['time', 'key']
        ).reset_index(level='key', drop=True)
    )
    df = df.reset_index()
    expected = (
        df[df.time.between(*(t.tz_convert(None) for t in context))]
        .sort_values(['key'])
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result_pd, expected)
