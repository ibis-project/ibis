import os

import pytest

import ibis

PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID', 'ibis-gbq')
DATASET_ID = 'testing'


def connect(project_id, dataset_id, application_name=None):
    ga = pytest.importorskip('google.auth')
    service_account = pytest.importorskip('google.oauth2.service_account')
    google_application_credentials = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", None
    )
    if google_application_credentials is None:
        pytest.skip(
            'Environment variable GOOGLE_APPLICATION_CREDENTIALS is '
            'not defined'
        )
    elif not google_application_credentials:
        pytest.skip(
            'Environment variable GOOGLE_APPLICATION_CREDENTIALS is empty'
        )
    elif not os.path.exists(google_application_credentials):
        pytest.skip(
            'Environment variable GOOGLE_APPLICATION_CREDENTIALS points '
            'to {}, which does not exist'.format(
                google_application_credentials
            )
        )

    skip_message = (
        'No BigQuery credentials found using project_id={}, '
        'dataset_id={}. Skipping BigQuery tests.'
    ).format(project_id, dataset_id)
    credentials = service_account.Credentials.from_service_account_file(
        google_application_credentials
    )
    try:
        return ibis.bigquery.connect(
            project_id,
            dataset_id,
            credentials=credentials,
            application_name=application_name,
        )
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip(skip_message)


@pytest.fixture(scope='session')
def project_id():
    return PROJECT_ID


@pytest.fixture(scope='session')
def client():
    return connect(PROJECT_ID, DATASET_ID)


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
