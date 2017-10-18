import six
import pytest

import pandas as pd
import ibis.util
from ibis.bigquery import client as ibc  # noqa: E402


pytest.importorskip('google.cloud.bigquery')
pytestmark = pytest.mark.bigquery


PROJECT_ID = 'bq-for-ibis'
DATASET_ID = 'ci_{}'.format(ibis.util.guid())
TABLE_ID = 'table_{}'.format(ibis.util.guid())


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
    json_file = six.BytesIO(
        '\n'.join((v.to_json() for (k, v) in df.iterrows())).encode('UTF-8')
    )
    schema = ibc.infer_schema_from_df(df)
    #
    bq_dataset = client._proxy.get_dataset(client.dataset_id)
    bq_dataset.create()
    bq_dataset.reload()
    #
    bq_table = bq_dataset.table(TABLE_ID, schema)
    bq_table.create()
    bq_table.reload()
    load_job = bq_table.upload_from_file(
        json_file, source_format='NEWLINE_DELIMITED_JSON',
    )
    # Wait for table load to complete.
    load_job.result()
    try:
        yield client
    finally:
        bq_table.delete()
        bq_dataset.delete()


@pytest.fixture(scope='session')
def table(client_with_table):
    return client_with_table.table(TABLE_ID)
