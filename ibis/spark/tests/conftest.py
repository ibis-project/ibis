import pytest

import ibis


@pytest.fixture(scope='session')
def client_empty():
    return ibis.spark.connect()


@pytest.fixture(scope='session')
def client_simple():
    client = ibis.spark.connect()
    df = client._session.createDataFrame([(1, 'a')], ['foo', 'bar'])
    df.createOrReplaceTempView('t')
    return client
