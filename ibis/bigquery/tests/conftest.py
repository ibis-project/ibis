import os

import pytest

import ibis


PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID', 'ibis-gbq')
DATASET_ID = 'testing'


def connect(project_id, dataset_id):
    ga = pytest.importorskip('google.auth')

    try:
        return ibis.bigquery.connect(project_id, dataset_id)
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip(
            'no BigQuery credentials found (project_id={}, dataset_id={}), '
            'skipping'.format(project_id, dataset_id)
        )


@pytest.fixture(scope='session')
def project_id():
    return PROJECT_ID


@pytest.fixture(scope='session')
def client():
    return connect(PROJECT_ID, DATASET_ID)


@pytest.fixture(scope='session')
def client_no_credentials():
    ga = pytest.importorskip('google.auth')

    try:
        return ibis.bigquery.connect(PROJECT_ID, DATASET_ID, credentials=None)
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip(
            'no BigQuery credentials found (project_id={}, dataset_id={}), '
            'skipping'.format(PROJECT_ID, DATASET_ID)
        )


@pytest.fixture(scope='session')
def client2():
    return connect(PROJECT_ID, DATASET_ID)


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


@pytest.fixture(scope='session')
def numeric_table(client):
    return client.table('numeric_table')


@pytest.fixture(scope='session')
def public():
    return connect(PROJECT_ID, dataset_id='bigquery-public-data.stackoverflow')
