import pytest

import pandas as pd
import ibis.util
bq = pytest.importorskip('google.datalab.bigquery')
from ibis.bigquery import client as ibc  # noqa: E402


PROJECT_ID = 'bq-for-ibis'
DATASET_ID = 'ci_{}'.format(ibis.util.guid())
TABLE = 'table_{}'.format(ibis.util.guid())


@pytest.fixture(scope='session')
def df():
    return pd.DataFrame(dict(
        int_column=[1], float_column=[2.],
        string_column=['a'], bool_column=[True],
    ))


@pytest.fixture(scope='session')
def client():
    return ibc.BigQueryClient(PROJECT_ID, DATASET_ID)


@pytest.fixture(scope='session')
def client_with_table(client, df):
    schema = bq.Schema.from_data(df)
    bq_dataset = client._proxy.get_dataset(client.dataset_id)
    bq_table = client._proxy.get_table(TABLE, client.dataset_id)
    assert not bq_dataset.exists()
    assert not bq_table.exists()
    #
    assert client.project_id == bq_dataset.name.project_id
    assert bq_dataset.name.project_id == bq_table._name_parts.project_id
    assert client.dataset_id == bq_dataset.name.dataset_id
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
