import pandas as pd
import pandas.util.testing as tm
import pyspark.sql.functions as F
import pytest
from pyspark.sql.window import Window

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


def test_time_indexed_window(client):
    table = client.table('time_indexed_table')
    window1 = ibis.trailing_window(
        preceding=ibis.interval(hours=1), order_by='time', group_by='key'
    )
    window2 = ibis.trailing_window(
        preceding=ibis.interval(hours=2), order_by='time', group_by='key'
    )
    result = table.mutate(
        mean_1h=table['value'].mean().over(window1),
        mean_2h=table['value'].mean().over(window2),
    ).compile()
    result_pd = result.toPandas()

    df = table.compile().toPandas()
    expected_win_1 = (
        df.set_index('time')
        .groupby('key')
        .value.rolling('1h', closed='both')
        .mean()
        .rename('mean_1h')
    ).reset_index(drop=True)
    expected_win_2 = (
        df.set_index('time')
        .groupby('key')
        .value.rolling('2h', closed='both')
        .mean()
        .rename('mean_2h')
    ).reset_index(drop=True)

    tm.assert_series_equal(result_pd['mean_1h'], expected_win_1)
    tm.assert_series_equal(result_pd['mean_2h'], expected_win_2)


def test_foward_looking_window(client):
    table = client.table('time_indexed_table')
    window = ibis.range_window(
        preceding=0, following=ibis.interval(hours=1), order_by='time'
    )
    result = table.mutate(
        mean_following_1h=table['value'].mean().over(window),
    ).compile()
    result_pd = result.toPandas()
    spark_window = (
        Window.partitionBy()
        .orderBy(F.col('time').cast('long'))
        .rangeBetween(0, 3600)
    )
    spark_table = table.compile()
    expected = spark_table.withColumn(
        'mean_following_1h', F.mean(spark_table['value']).over(spark_window),
    ).toPandas()
    tm.assert_frame_equal(result_pd, expected)
