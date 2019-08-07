import pandas as pd
import pandas.util.testing as tm
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

import ibis


@pytest.fixture(scope='session')
def client():
    session = SparkSession.builder.getOrCreate()
    client = ibis.pyspark.connect(session)
    df = client._session.range(0, 10)
    df = df.withColumn("str_col", F.lit('value'))
    df.createTempView('table1')

    df1 = client._session.createDataFrame([(True,), (False,)]).toDF('v')
    df1.createTempView('table2')
    return client


def test_basic(client):
    table = client.table('table1')
    result = table.compile().toPandas()
    expected = pd.DataFrame({'id': range(0, 10), 'str_col': 'value'})

    tm.assert_frame_equal(result, expected)


def test_projection(client):
    table = client.table('table1')
    result1 = table.mutate(v=table['id']).compile().toPandas()

    expected1 = pd.DataFrame(
        {
            'id': range(0, 10),
            'str_col': 'value',
            'v': range(0, 10)
        }
    )

    result2 = (
        table.mutate(v=table['id']).mutate(v2=table['id'])
        .compile().toPandas()
    )

    expected2 = pd.DataFrame(
        {
            'id': range(0, 10),
            'str_col': 'value',
            'v': range(0, 10),
            'v2': range(0, 10)
        }
    )

    tm.assert_frame_equal(result1, expected1)
    tm.assert_frame_equal(result2, expected2)


def test_aggregation_col(client):
    table = client.table('table1')
    result = table['id'].count().execute()
    assert result == table.compile().count()


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
    result = (
        table
        .mutate(
            grouped_demeaned=table['id'] - table['id'].mean().over(w))
        .compile()
    )
    result2 = (
        table
        .groupby('id')
        .mutate(
            grouped_demeaned=table['id'] - table['id'].mean())
        .compile()
    )

    spark_window = Window.partitionBy()
    spark_table = table.compile()
    expected = spark_table.withColumn(
        'grouped_demeaned',
        spark_table['id'] - F.mean(spark_table['id']).over(spark_window)
    )

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())
    tm.assert_frame_equal(result2.toPandas(), expected.toPandas())


def test_greatest(client):
    table = client.table('table1')
    result = (
        table
        .mutate(greatest=ibis.greatest(table.id))
        .compile()
    )
    df = table.compile()
    expected = table.compile().withColumn('greatest', df.id)

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_selection(client):
    table = client.table('table1')
    table = table.mutate(id2=table['id'])

    result1 = table[['id']].compile()
    result2 = table[['id', 'id2']].compile()
    result3 = table[[table, (table.id + 1).name('plus1')]].compile()
    result4 = table[[(table.id + 1).name('plus1'), table]].compile()

    df = table.compile()
    tm.assert_frame_equal(result1.toPandas(), df[['id']].toPandas())
    tm.assert_frame_equal(result2.toPandas(), df[['id', 'id2']].toPandas())
    tm.assert_frame_equal(result3.toPandas(),
                          df[[df.columns]].withColumn('plus1', df.id + 1)
                          .toPandas())
    tm.assert_frame_equal(result4.toPandas(),
                          df.withColumn('plus1', df.id + 1)
                          [['plus1', *df.columns]].toPandas())


def test_join(client):
    table = client.table('table1')
    result = table.join(table, 'id').compile()
    spark_table = table.compile()
    expected = (
        spark_table
        .join(spark_table, spark_table['id'] == spark_table['id'])
    )

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())
