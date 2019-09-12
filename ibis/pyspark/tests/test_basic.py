import pandas as pd
import pandas.util.testing as tm
import pytest
from pytest import param

import ibis

pytest.importorskip('pyspark')
pytestmark = pytest.mark.pyspark


@pytest.fixture(scope='session')
def client():
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

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
        {'id': range(0, 10), 'str_col': 'value', 'v': range(0, 10)}
    )

    result2 = (
        table.mutate(v=table['id'])
        .mutate(v2=table['id'])
        .mutate(id=table['id'] * 2)
        .compile()
        .toPandas()
    )

    expected2 = pd.DataFrame(
        {
            'id': range(0, 20, 2),
            'str_col': 'value',
            'v': range(0, 10),
            'v2': range(0, 10),
        }
    )

    tm.assert_frame_equal(result1, expected1)
    tm.assert_frame_equal(result2, expected2)


def test_aggregation_col(client):
    table = client.table('table1')
    result = table['id'].count().execute()
    assert result == table.compile().count()


def test_aggregation(client):
    import pyspark.sql.functions as F

    table = client.table('table1')
    result = table.aggregate(table['id'].max()).compile()
    expected = table.compile().agg(F.max('id'))

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_groupby(client):
    import pyspark.sql.functions as F

    table = client.table('table1')
    result = table.groupby('id').aggregate(table['id'].max()).compile()
    expected = table.compile().groupby('id').agg(F.max('id'))

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_window(client):
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window

    table = client.table('table1')
    w = ibis.window()
    result = table.mutate(
        grouped_demeaned=table['id'] - table['id'].mean().over(w)
    ).compile()

    spark_window = Window.partitionBy()
    spark_table = table.compile()
    expected = spark_table.withColumn(
        'grouped_demeaned',
        spark_table['id'] - F.mean(spark_table['id']).over(spark_window),
    )

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_greatest(client):
    table = client.table('table1')
    result = table.mutate(greatest=ibis.greatest(table.id)).compile()
    df = table.compile()
    expected = table.compile().withColumn('greatest', df.id)

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_selection(client):
    table = client.table('table1')
    table = table.mutate(id2=table['id'] * 2)

    result1 = table[['id']].compile()
    result2 = table[['id', 'id2']].compile()
    result3 = table[[table, (table.id + 1).name('plus1')]].compile()
    result4 = table[[(table.id + 1).name('plus1'), table]].compile()

    df = table.compile()
    tm.assert_frame_equal(result1.toPandas(), df[['id']].toPandas())
    tm.assert_frame_equal(result2.toPandas(), df[['id', 'id2']].toPandas())
    tm.assert_frame_equal(
        result3.toPandas(),
        df[[df.columns]].withColumn('plus1', df.id + 1).toPandas(),
    )
    tm.assert_frame_equal(
        result4.toPandas(),
        df.withColumn('plus1', df.id + 1)[['plus1', *df.columns]].toPandas(),
    )


def test_join(client):
    table = client.table('table1')
    result = table.join(table, ['id', 'str_col']).compile()
    spark_table = table.compile()
    expected = spark_table.join(spark_table, ['id', 'str_col'])

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


@pytest.mark.parametrize(
    ('filter_fn', 'expected_fn'),
    [
        param(lambda t: t.filter(t.id < 5), lambda df: df[df.id < 5]),
        param(lambda t: t.filter(t.id != 5), lambda df: df[df.id != 5]),
        param(
            lambda t: t.filter([t.id < 5, t.str_col == 'na']),
            lambda df: df[df.id < 5][df.str_col == 'na'],
        ),
        param(
            lambda t: t.filter((t.id > 3) & (t.id < 11)),
            lambda df: df[(df.id > 3) & (df.id < 11)],
        ),
        param(
            lambda t: t.filter((t.id == 3) | (t.id == 5)),
            lambda df: df[(df.id == 3) | (df.id == 5)],
        ),
    ],
)
def test_filter(client, filter_fn, expected_fn):
    table = client.table('table1')

    result = filter_fn(table).compile()

    df = table.compile()
    expected = expected_fn(df)

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())
