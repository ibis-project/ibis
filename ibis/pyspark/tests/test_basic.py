import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis
import pyspark.sql.functions as F
from pyspark.sql.window import Window

@pytest.fixture(scope='session')
def client():
    client = ibis.pyspark.connect()
    df = client._session.range(0, 10)
    df.createTempView('table1')
    return client


def test_basic(client):
    table = client.table('table1')
    result = table.compile().toPandas()
    expected = pd.DataFrame({'id': range(0, 10)})

    tm.assert_frame_equal(result, expected)


def test_projection(client):
    table = client.table('table1')
    result1 = table.mutate(v=table['id']).compile().toPandas()

    expected1 = pd.DataFrame(
        {
            'id': range(0, 10),
            'v': range(0, 10)
        }
    )

    result2 = table.mutate(v=table['id']).mutate(v2=table['id']).compile().toPandas()

    expected2 = pd.DataFrame(
        {
            'id': range(0, 10),
            'v': range(0, 10),
            'v2': range(0, 10)
        }
    )

    tm.assert_frame_equal(result1, expected1)
    tm.assert_frame_equal(result2, expected2)


def test_aggregation(client):
    table = client.table('table1')
    result = table.aggregate(table['id'].max()).compile()
    expected = table.compile().agg(F.max('id'))

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_groupby(client):
    table = client.table('table1')
    result = table.groupby('id').aggregate(table['id'].max()).compile()
    expected = table.compile().groupby('id').agg(F.max('id'))

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_window(client):
    table = client.table('table1')
    w = ibis.window()
    result = table.mutate(grouped_demeaned = table['id'] - table['id'].mean().over(w)).compile()
    result2 = table.groupby('id').mutate(grouped_demeaned = table['id'] - table['id'].mean()).compile()

    spark_window = Window.partitionBy()
    spark_table = table.compile()
    expected = spark_table.withColumn(
        'grouped_demeaned',
        spark_table['id'] - F.mean(spark_table['id']).over(spark_window)
    )

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())
    tm.assert_frame_equal(result2.toPandas(), expected.toPandas())
