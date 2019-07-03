import pytest

import ibis


@pytest.fixture(scope='session')
def client():
    pytest.importorskip('pyspark')
    client = ibis.spark.connect()

    df_simple = client._session.createDataFrame([(1, 'a')], ['foo', 'bar'])
    df_simple.createOrReplaceTempView('simple')

    df_struct = client._session.createDataFrame(
        [((1, 2, 'a'),)],
        ['struct_col']
    )
    df_struct.createOrReplaceTempView('struct')

    df_nested_types = client._session.createDataFrame(
        [
            (
                [1, 2],
                [[3, 4], [5, 6]],
                {'a' : [[2, 4], [3, 5]]},
            )
        ],
        [
            'list_of_ints',
            'list_of_list_of_ints',
            'map_string_list_of_list_of_ints'
        ]
    )
    df_nested_types.createOrReplaceTempView('nested_types')

    df_complicated = client._session.createDataFrame(
        [({(1, 3) : [[2, 4], [3, 5]]},)],
        ['map_tuple_list_of_list_of_ints']
    )
    df_complicated.createOrReplaceTempView('complicated')

    return client


@pytest.fixture(scope='session')
def simple(client):
    return client.table('simple')


@pytest.fixture(scope='session')
def struct(client):
    return client.table('struct')


@pytest.fixture(scope='session')
def nested_types(client):
    return client.table('nested_types')


@pytest.fixture(scope='session')
def complicated(client):
    return client.table('complicated')
