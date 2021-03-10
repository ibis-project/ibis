from unittest import mock

import pydata_google_auth
import pytest
from google.auth import credentials as auth
from google.cloud import bigquery as bq

import ibis
from ibis.backends import bigquery as bq_backend

pytestmark = pytest.mark.bigquery


def test_repeated_project_name(project_id, credentials):
    con = bq_backend.connect(
        project_id=project_id,
        dataset_id='{}.testing'.format(project_id),
        credentials=credentials,
    )
    assert 'functional_alltypes' in con.list_tables()


def test_project_id_different_from_default_credentials(monkeypatch):
    creds = mock.create_autospec(auth.Credentials)

    def mock_credentials(*args, **kwargs):
        return creds, 'default-project-id'

    monkeypatch.setattr(pydata_google_auth, 'default', mock_credentials)
    con = bq_backend.connect(project_id='explicit-project-id',)
    assert con.billing_project == 'explicit-project-id'


def test_without_dataset(project_id, credentials):
    con = bq_backend.connect(
        project_id=project_id, dataset_id=None, credentials=credentials,
    )
    with pytest.raises(ValueError, match="Unable to determine BigQuery"):
        con.list_tables()


def test_application_name_sets_user_agent(
    project_id, credentials, monkeypatch
):
    mock_client = mock.create_autospec(bq.Client)
    monkeypatch.setattr(bq, 'Client', mock_client)
    bq_backend.connect(
        project_id=project_id,
        dataset_id='bigquery-public-data.stackoverflow',
        application_name='my-great-app/0.7.0',
        credentials=credentials,
    )
    info = mock_client.call_args[1]['client_info']
    user_agent = info.to_user_agent()
    assert ' ibis/{}'.format(ibis.__version__) in user_agent
    assert 'my-great-app/0.7.0 ' in user_agent
