import pytest

import ibis


@pytest.fixture(scope='session')
def client():
    pytest.importorskip('pyspark')
    client = ibis.spark.connect()
    df = client._session.createDataFrame([(1, 'a')], ['foo', 'bar'])
    df.createOrReplaceTempView('simple')
    df1 = client._session.createDataFrame(
        [
            (
                [1, 2],
                [[3, 4], [5, 6]],
                {(1, 3) : [[2, 4], [3, 5]]},
            )
        ],
        ['c1', 'c2', 'c3']
    )
    df1.createOrReplaceTempView('nested_types')
    return client


@pytest.fixture(scope='session')
def simple(client):
    return client.table('simple')


@pytest.fixture(scope='session')
def nested_types(client):
    return client.table('nested_types')
