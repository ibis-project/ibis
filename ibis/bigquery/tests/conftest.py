import uuid
import pytest

import ibis
import pandas as pd
from ibis.bigquery import client as ibc
import google.datalab.bigquery as bq


PROJECT_ID = 'bq-for-ibis'
DATASET_ID = 'ci_{}'.format(uuid.uuid4()).replace('-', '_')
TABLE = 'table_{}'.format(uuid.uuid4()).replace('-', '_')



@pytest.fixture(scope='session')
def bq_context():
    return ibc._bq_get_context(PROJECT_ID)


@pytest.fixture(scope='session')
def bq_datasets(bq_context):
    return bq.Datasets(bq_context)


@pytest.fixture(scope='session')
def bq_dataset(bq_context):
    return bq.Dataset(DATASET_ID, context=bq_context)


@pytest.fixture(scope='session')
def bq_table(bq_dataset):
    name = bq_dataset.name.dataset_id + '.' + TABLE
    # can't use client.table: it needs to exist so we can get a schema
    return bq.Table(name, context=bq_dataset._context)


@pytest.fixture(scope='session')
def df():
    return pd.DataFrame(dict(
        int_column=[1], float_column=[2.],
        string_column=['a'], bool_column=[True],
    ))


@pytest.fixture(scope='session')
def client(bq_dataset):
    bq_name = bq_dataset.name
    client = ibc.BigQueryClient(bq_name.project_id, bq_name.dataset_id)
    ibis.options.default_backend = client
    return client


@pytest.fixture(scope='session')
def client_with_table(client, bq_table, df):
    bq_dataset = client._dataset
    schema = bq.Schema.from_data(df)
    assert not bq_dataset.exists()
    assert not bq_table.exists()
    #
    assert client._project_id == bq_dataset.name.project_id
    assert bq_dataset.name.project_id == bq_table._name_parts.project_id
    assert client._dataset_id == bq_dataset.name.dataset_id
    assert bq_dataset.name.dataset_id == bq_table._name_parts.dataset_id
    #
    bq_dataset.create()
    bq_table.create(schema)
    bq_table.insert(df)
    assert bq_dataset.exists()
    assert bq_table.exists()
    _df = bq_table.to_dataframe()
    assert _df.equals(df)
    yield client
    bq_table.delete()
    bq_dataset.delete()


@pytest.fixture(scope='session')
def table(client_with_table):
    return client_with_table.table(TABLE)
