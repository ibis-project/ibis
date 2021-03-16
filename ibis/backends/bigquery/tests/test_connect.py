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


def test_auth_default(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    bq_backend.connect(
        project_id=project_id, dataset_id='bigquery-public-data.stackoverflow',
    )

    assert len(mock_calls) == 1
    args, kwargs = mock_calls[0]
    assert len(args) == 1
    scopes = args[0]
    assert scopes == bq_backend.SCOPES
    auth_local_webserver = kwargs["use_local_webserver"]
    auth_cache = kwargs["credentials_cache"]
    assert not auth_local_webserver
    assert isinstance(
        auth_cache, pydata_google_auth.cache.ReadWriteCredentialsCache,
    )


def test_auth_local_webserver(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    bq_backend.connect(
        project_id=project_id,
        dataset_id='bigquery-public-data.stackoverflow',
        auth_local_webserver=True,
    )

    assert len(mock_calls) == 1
    _, kwargs = mock_calls[0]
    auth_local_webserver = kwargs["use_local_webserver"]
    assert auth_local_webserver


def test_auth_external_data(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    bq_backend.connect(
        project_id=project_id,
        dataset_id='bigquery-public-data.stackoverflow',
        auth_external_data=True,
    )

    assert len(mock_calls) == 1
    args, _ = mock_calls[0]
    assert len(args) == 1
    scopes = args[0]
    assert scopes == bq_backend.EXTERNAL_DATA_SCOPES


def test_auth_cache_reauth(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    bq_backend.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        auth_cache="reauth",
    )

    assert len(mock_calls) == 1
    _, kwargs = mock_calls[0]
    auth_cache = kwargs["credentials_cache"]
    assert isinstance(
        auth_cache, pydata_google_auth.cache.WriteOnlyCredentialsCache,
    )


def test_auth_cache_none(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    bq_backend.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        auth_cache="none",
    )

    assert len(mock_calls) == 1
    _, kwargs = mock_calls[0]
    auth_cache = kwargs["credentials_cache"]
    assert auth_cache is pydata_google_auth.cache.NOOP


def test_auth_cache_unknown(project_id):
    with pytest.raises(ValueError, match="unexpected value for auth_cache"):
        bq_backend.connect(
            project_id=project_id,
            dataset_id="bigquery-public-data.stackoverflow",
            auth_cache="not_a_real_cache",
        )
