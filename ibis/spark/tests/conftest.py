import pytest

from ibis.tests.all.conftest import get_spark_testing_client


@pytest.fixture(scope='session')
def client(data_directory):
    pytest.importorskip('pyspark')
    return get_spark_testing_client(data_directory)


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
