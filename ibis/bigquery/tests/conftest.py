import os

import pytest

import ibis


PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID')
DATASET_ID = 'testing'


@pytest.fixture(scope='session')
def client():
    return ibis.bigquery.connect(PROJECT_ID, DATASET_ID)


@pytest.fixture(scope='session')
def alltypes(client):
    return client.table('functional_alltypes')


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def struct_table(client):
    return client.table('struct_table')
