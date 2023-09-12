from __future__ import annotations

from unittest import mock

import google.api_core.client_options
import google.api_core.exceptions as gexc
import pydata_google_auth
import pytest
from google.auth import credentials as auth
from google.cloud import bigquery as bq
from google.cloud import bigquery_storage_v1 as bqstorage

import ibis


def test_repeated_project_name(project_id, credentials):
    con = ibis.bigquery.connect(
        project_id=project_id,
        dataset_id=f"{project_id}.testing",
        credentials=credentials,
    )
    try:
        assert "functional_alltypes" in con.list_tables()
    except gexc.Forbidden:
        pytest.skip("Cannot access BigQuery")


def test_project_id_different_from_default_credentials(monkeypatch):
    creds = mock.create_autospec(auth.Credentials)

    def mock_credentials(*args, **kwargs):
        return creds, "default-project-id"

    monkeypatch.setattr(pydata_google_auth, "default", mock_credentials)
    con = ibis.bigquery.connect(
        project_id="explicit-project-id",
    )
    assert con.billing_project == "explicit-project-id"


def test_without_dataset(project_id, credentials):
    con = ibis.bigquery.connect(project_id=project_id, credentials=credentials)
    with pytest.raises(ValueError, match="Unable to determine BigQuery"):
        con.list_tables()


def test_application_name_sets_user_agent(project_id, credentials, monkeypatch):
    mock_client = mock.create_autospec(bq.Client)
    monkeypatch.setattr(bq, "Client", mock_client)
    ibis.bigquery.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        application_name="my-great-app/0.7.0",
        credentials=credentials,
    )
    info = mock_client.call_args[1]["client_info"]
    user_agent = info.to_user_agent()
    assert f" ibis/{ibis.__version__}" in user_agent
    assert "my-great-app/0.7.0 " in user_agent


def test_auth_default(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    ibis.bigquery.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
    )

    assert len(mock_calls) == 1
    args, kwargs = mock_calls[0]
    assert len(args) == 1
    scopes = args[0]
    assert scopes == ibis.backends.bigquery.SCOPES
    auth_local_webserver = kwargs["use_local_webserver"]
    auth_cache = kwargs["credentials_cache"]
    assert auth_local_webserver
    assert isinstance(
        auth_cache,
        pydata_google_auth.cache.ReadWriteCredentialsCache,
    )


def test_auth_local_webserver(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    ibis.bigquery.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        auth_local_webserver=False,
    )

    assert len(mock_calls) == 1
    _, kwargs = mock_calls[0]
    auth_local_webserver = kwargs["use_local_webserver"]
    assert not auth_local_webserver


def test_auth_external_data(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    ibis.bigquery.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        auth_external_data=True,
    )

    assert len(mock_calls) == 1
    args, _ = mock_calls[0]
    assert len(args) == 1
    scopes = args[0]
    assert scopes == ibis.backends.bigquery.EXTERNAL_DATA_SCOPES


def test_auth_cache_reauth(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    ibis.bigquery.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        auth_cache="reauth",
    )

    assert len(mock_calls) == 1
    _, kwargs = mock_calls[0]
    auth_cache = kwargs["credentials_cache"]
    assert isinstance(
        auth_cache,
        pydata_google_auth.cache.WriteOnlyCredentialsCache,
    )


def test_auth_cache_none(project_id, credentials, monkeypatch):
    mock_calls = []

    def mock_default(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return credentials, project_id

    monkeypatch.setattr(pydata_google_auth, "default", mock_default)

    ibis.bigquery.connect(
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
        ibis.bigquery.connect(
            project_id=project_id,
            dataset_id="bigquery-public-data.stackoverflow",
            auth_cache="not_a_real_cache",
        )


def test_client_with_regional_endpoints(
    project_id, credentials, dataset_id, dataset_id_tokyo, region_tokyo
):
    bq_options = google.api_core.client_options.ClientOptions(
        api_endpoint=f"https://{region_tokyo}-bigquery.googleapis.com"
    )
    bq_client = bq.Client(
        client_options=bq_options, project=project_id, credentials=credentials
    )

    # Note there is no protocol specifier for gRPC APIs.
    bqstorage_options = google.api_core.client_options.ClientOptions(
        api_endpoint=f"{region_tokyo}-bigquerystorage.googleapis.com"
    )
    bqstorage_client = bqstorage.BigQueryReadClient(
        client_options=bqstorage_options, credentials=credentials
    )

    con = ibis.bigquery.connect(
        client=bq_client, storage_client=bqstorage_client, project_id=project_id
    )

    # Fails because dataset not in Tokyo.
    with pytest.raises(gexc.NotFound, match=dataset_id):
        con.table(f"{dataset_id}.functional_alltypes")

    # Succeeds because dataset is in Tokyo.
    alltypes = con.table(f"{dataset_id_tokyo}.functional_alltypes")
    df = alltypes.limit(2).execute()
    assert len(df.index) == 2
