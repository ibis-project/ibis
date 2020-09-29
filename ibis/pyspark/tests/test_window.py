import pandas.util.testing as tm
import pyspark.sql.functions as F
import pytest
from pyspark.sql.window import Window

import ibis

pytest.importorskip('pyspark')
pytestmark = pytest.mark.pyspark


def test_time_indexed_window(client):
    table = client.table('time_indexed_table')
    window1 = ibis.trailing_window(
        preceding=ibis.interval(hours=1), order_by='time', group_by='key'
    )
    result = table.mutate(
        mean_1h=table['value'].mean().over(window1)
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
    tm.assert_series_equal(result_pd['mean_1h'], expected_win_1)


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
