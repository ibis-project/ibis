import os

import pytest

import ibis


PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID', 'ibis-gbq')
DATASET_ID = 'testing'


@pytest.fixture(scope='session')
def client():
    ga = pytest.importorskip('google.auth')

    try:
        return ibis.bigquery.connect(PROJECT_ID, DATASET_ID)
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip("no credentials found, skipping")


@pytest.fixture(scope='session')
def client2():
    ga = pytest.importorskip('google.auth')

    try:
        return ibis.bigquery.connect(PROJECT_ID, DATASET_ID)
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip("no credentials found, skipping")


@pytest.fixture(scope='session')
def alltypes(client):
    return client.table('functional_alltypes')


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def parted_alltypes(client):
    return client.table('functional_alltypes_parted')


@pytest.fixture(scope='session')
def parted_df(parted_alltypes):
    return parted_alltypes.execute()


@pytest.fixture(scope='session')
def struct_table(client):
    return client.table('struct_table')
