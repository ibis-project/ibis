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
import ibis.common.exceptions as exc
from ibis.util import gen_name


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

    def mock_credentials(*_, **__):
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


def test_client_with_regional_endpoints(project_id, credentials, dataset_id):
    region_tokyo = "asia-northeast1"
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

    tokyo_con = ibis.bigquery.connect(
        client=bq_client, storage_client=bqstorage_client, project_id=project_id
    )

    tokyo_con.raw_sql(
        f"""
        CREATE SCHEMA IF NOT EXISTS {dataset_id}_tokyo OPTIONS (
            location = 'asia-northeast1'
        );

        CREATE OR REPLACE TABLE {dataset_id}_tokyo.functional_alltypes (
            id INT64,
            bool_col BOOLEAN,
            tinyint_col INT64,
            smallint_col INT64,
            int_col INT64,
            bigint_col INT64,
            float_col FLOAT64,
            double_col FLOAT64,
            date_string_col STRING,
            string_col STRING,
            timestamp_col DATETIME,
            year INT64,
            month INT64
        )
        """
    )

    # Fails because dataset not in Tokyo.
    with pytest.raises(exc.TableNotFound, match=dataset_id):
        tokyo_con.table(f"{dataset_id}.functional_alltypes")

    # Succeeds because dataset is in Tokyo.
    alltypes = tokyo_con.table(f"{dataset_id}_tokyo.functional_alltypes")
    df = alltypes.execute()
    assert df.empty
    assert not len(alltypes.to_pyarrow())


def test_create_table_from_memtable_needs_quotes(project_id, dataset_id, credentials):
    con = ibis.bigquery.connect(
        project_id=project_id, dataset_id=dataset_id, credentials=credentials
    )

    name = gen_name("region-table")
    schema = dict(its_always="str", quoting="int")

    t = con.create_table(name, schema=schema)

    try:
        assert t.schema() == ibis.schema(schema)
    finally:
        con.drop_table(name)


def test_project_id_from_arg(project_id):
    con = ibis.bigquery.connect(project_id=project_id)
    assert con.project_id == project_id


def test_project_id_from_client(project_id):
    bq_client = bq.Client(project=project_id)
    con = ibis.bigquery.connect(client=bq_client, project_id="not-a-real-project")
    assert con.project_id == project_id


def test_project_id_from_default(default_credentials):
    _, default_project_id = default_credentials
    # `connect()` re-evaluates default credentials and sets project_id since no client nor explicit project_id is provided
    con = ibis.bigquery.connect()
    assert con.project_id == default_project_id


def test_project_id_missing(credentials):
    with pytest.raises(ValueError, match="Project ID could not be identified.*"):
        ibis.bigquery.connect(credentials=credentials)
