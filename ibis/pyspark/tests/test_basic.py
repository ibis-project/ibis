import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis

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
    import ipdb; ipdb.set_trace()
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


def test_udf(client):
    table = client.table('table1')
