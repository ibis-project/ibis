import pandas.util.testing as tm
import pyspark.sql.functions as F
import pytest
from pyspark.sql.window import Window
from pytest import param

import ibis

pytest.importorskip('pyspark')
pytestmark = pytest.mark.pyspark


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
        ),
        param(
            ibis.trailing_window(
                preceding=ibis.interval(hours=2),
                order_by='time',
                group_by='key',
            ),
            (-7200, 0),
        ),
        param(
            ibis.range_window(
                preceding=0,
                following=ibis.interval(hours=1),
                order_by='time',
                group_by='key',
            ),
            (0, 3600),
        ),
    ],
)
def test_time_indexed_window(client, ibis_window, spark_range):
    table = client.table('time_indexed_table')
    result = table.mutate(
        mean=table['value'].mean().over(ibis_window)
    ).compile()
    result_pd = result.toPandas()
    spark_table = table.compile()
    spark_window = (
        Window.partitionBy('key')
        .orderBy(F.col('time').cast('long'))
        .rangeBetween(*spark_range)
    )
    expected = spark_table.withColumn(
        'mean', F.mean(spark_table['value']).over(spark_window),
    ).toPandas()
    tm.assert_frame_equal(result_pd, expected)


# TODO: multi windows don't update scope correctly
@pytest.mark.xfail(reason='Issue #2412', strict=True)
def test_multiple_windows(client):
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
