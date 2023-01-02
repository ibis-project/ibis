from __future__ import annotations

import os

import google.auth
import google.auth.exceptions
import pytest

import ibis

DEFAULT_PROJECT_ID = "ibis-gbq"
PROJECT_ID_ENV_VAR = "GOOGLE_BIGQUERY_PROJECT_ID"
DATASET_ID = "ibis_gbq_testing"


def pytest_addoption(parser):
    parser.addoption(
        "--save-dataset",
        action="store_true",
        default=False,
        help="saves all test data in the testing dataset",
    )
    parser.addoption(
        "--no-refresh-dataset",
        action="store_true",
        default=False,
        help="do not refresh the test data in the testing dataset",
    )


@pytest.fixture(scope="session")
def dataset_id() -> str:
    return DATASET_ID


@pytest.fixture(scope="session")
def default_credentials():
    try:
        credentials, project_id = google.auth.default(
            scopes=ibis.backends.bigquery.EXTERNAL_DATA_SCOPES
        )
    except google.auth.exceptions.DefaultCredentialsError as exc:
        pytest.skip(f"Could not get GCP credentials: {exc}")

    return credentials, project_id


@pytest.fixture(scope="session")
def project_id(default_credentials):
    return (
        os.environ.get(PROJECT_ID_ENV_VAR, default_credentials[1]) or DEFAULT_PROJECT_ID
    )


@pytest.fixture(scope="session")
def credentials(default_credentials):
    credentials, _ = default_credentials
    return credentials


@pytest.fixture(scope="session")
def client(credentials, project_id, dataset_id):
    return ibis.bigquery.connect(
        project_id=project_id, dataset_id=dataset_id, credentials=credentials
    )


@pytest.fixture(scope="session")
def client2(credentials, project_id, dataset_id):
    return ibis.bigquery.connect(
        project_id=project_id, dataset_id=dataset_id, credentials=credentials
    )


@pytest.fixture(scope="session")
def alltypes(client):
    return client.table("functional_alltypes")


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def parted_alltypes(client):
    return client.table("functional_alltypes_parted")


@pytest.fixture(scope="session")
def struct_table(client):
    return client.table("struct_table")


@pytest.fixture(scope="session")
def numeric_table(client):
    return client.table("numeric_table")


@pytest.fixture(scope="session")
def public(project_id, credentials):
    return ibis.bigquery.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        credentials=credentials,
    )
