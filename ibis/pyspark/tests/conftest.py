import pytest

import ibis


@pytest.fixture(scope='session')
def client():
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    session = SparkSession.builder.getOrCreate()
    client = ibis.pyspark.connect(session)

    df = client._session.range(0, 10)
    df = df.withColumn("str_col", F.lit('value'))
    df.createTempView('basic_table')

    df_dates = client._session.createDataFrame(
        [['2018-01-02'], ['2018-01-03'], ['2018-01-04']],
        ['date_str']
    )
    df_dates.createTempView('date_table')

    df_arrays = client._session.createDataFrame(
        [
            ['k1', [1, 2, 3], ['a']],
            ['k2', [4, 5], ['test1', 'test2', 'test3']],
            ['k3', [6], ['w', 'x', 'y', 'z']],
            ['k1', [], ['cat', 'dog']],
            ['k1', [7, 8], []]
        ],
        ['key', 'array_int', 'array_str']
    )
    df_arrays.createTempView('array_table')
    return client
