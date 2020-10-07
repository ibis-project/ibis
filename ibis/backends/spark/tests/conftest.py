import os

import pytest

import ibis.util as util
from ibis.tests.all.conftest import get_spark_testing_client


@pytest.fixture(scope='session', autouse=True)
def test_data_db(client):
    try:
        name = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')
        client.create_database(name)
        client.set_database(name)
        yield name
    finally:
        client.drop_database(name, force=True)


@pytest.fixture(scope='session')
def client(data_directory):
    pytest.importorskip('pyspark')
    return get_spark_testing_client(data_directory)


@pytest.fixture(scope='session')
def con(client):
    # some of the tests use `con` instead of `client`
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


@pytest.fixture(scope='session')
def alltypes(client):
    return client.table('functional_alltypes').relabel(
        {'Unnamed: 0': 'Unnamed:0'}
    )


@pytest.fixture(scope='session')
def alltypes_df(alltypes):
    return alltypes.execute()


def _random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


@pytest.fixture(scope='session')
def tmp_dir():
    return '/tmp/__ibis_test_{}'.format(util.guid())


@pytest.fixture
def temp_database(con, test_data_db):
    name = _random_identifier('database')
    con.create_database(name)
    try:
        yield name
    finally:
        con.set_database(test_data_db)
        con.drop_database(name, force=True)


@pytest.fixture
def temp_table(con):
    name = _random_identifier('table')
    try:
        yield name
    finally:
        assert con.exists_table(name), name
        con.drop_table(name)


@pytest.fixture
def temp_table_db(con, temp_database):
    name = _random_identifier('table')
    try:
        yield temp_database, name
    finally:
        assert con.exists_table(name, database=temp_database), name
        con.drop_table(name, database=temp_database)


@pytest.fixture
def temp_view(con):
    name = _random_identifier('view')
    try:
        yield name
    finally:
        assert con.exists_table(name), name
        con.drop_view(name)


@pytest.fixture
def temp_view_db(con, temp_database):
    name = _random_identifier('view')
    try:
        yield temp_database, name
    finally:
        assert con.exists_table(name, database=temp_database), name
        con.drop_view(name, database=temp_database)
