import six
import pytest

import pandas as pd

import ibis


PROJECT_ID = 'ibis-gbq'
DATASET_ID = 'testing'
TABLE_ID = 'functional_alltypes'


@pytest.fixture(scope='session')
def client():
    return ibis.bigquery.connect(PROJECT_ID, DATASET_ID)


@pytest.fixture(scope='session')
def alltypes(client):
    return client.table(TABLE_ID)


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def table_id():
    return TABLE_ID
