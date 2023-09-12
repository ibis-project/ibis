from __future__ import annotations

import os

import google.api_core.exceptions as gexc
import google.auth
import google.auth.exceptions
import pytest

import ibis

DEFAULT_PROJECT_ID = "ibis-gbq"
PROJECT_ID_ENV_VAR = "GOOGLE_BIGQUERY_PROJECT_ID"
DATASET_ID = "ibis_gbq_testing"
DATASET_ID_TOKYO = "ibis_gbq_testing_tokyo"
REGION_TOKYO = "asia-northeast1"


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
def dataset_id_tokyo() -> str:
    return DATASET_ID_TOKYO


@pytest.fixture(scope="session")
def region_tokyo() -> str:
    return REGION_TOKYO


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
def con(credentials, project_id, dataset_id):
    con = ibis.bigquery.connect(
        project_id=project_id, dataset_id=dataset_id, credentials=credentials
    )
    try:
        con.sql("SELECT 1")
    except gexc.Forbidden:
        pytest.skip("Cannot access BigQuery")
    else:
        return con


@pytest.fixture(scope="session")
def con2(credentials, project_id, dataset_id):
    con = ibis.bigquery.connect(
        project_id=project_id, dataset_id=dataset_id, credentials=credentials
    )
    try:
        con.sql("SELECT 1")
    except gexc.Forbidden:
        pytest.skip("Cannot access BigQuery")
    else:
        return con


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def parted_alltypes(con):
    return con.table("functional_alltypes_parted")


@pytest.fixture(scope="session")
def struct_table(con):
    return con.table("struct_table")


@pytest.fixture(scope="session")
def numeric_table(con):
    return con.table("numeric_table")


@pytest.fixture(scope="session")
def public(project_id, credentials):
    con = ibis.bigquery.connect(
        project_id=project_id,
        dataset_id="bigquery-public-data.stackoverflow",
        credentials=credentials,
    )
    try:
        con.sql("SELECT 1")
    except gexc.Forbidden:
        pytest.skip("Cannot access BigQuery")
    else:
        return con
