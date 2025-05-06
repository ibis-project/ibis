from __future__ import annotations

import getpass
import os
from typing import Any

import google.api_core.exceptions as gexc
import google.auth
import pytest
from google.cloud import bigquery as bq

import ibis
from ibis.backends.bigquery import EXTERNAL_DATA_SCOPES, Backend
from ibis.backends.tests.base import PYTHON_SHORT_VERSION, BackendTest

DATASET_ID = f"ibis_gbq_testing_{getpass.getuser()}_{PYTHON_SHORT_VERSION}"
DEFAULT_PROJECT_ID = "ibis-gbq"
PROJECT_ID_ENV_VAR = "GOOGLE_BIGQUERY_PROJECT_ID"


class TestConf(BackendTest):
    """Backend-specific class with information for testing."""

    returned_timestamp_unit = "us"
    supports_structs = True
    supports_json = True
    check_names = False
    force_sort = True
    deps = ("google.cloud.bigquery",)

    @staticmethod
    def format_table(name: str) -> str:
        return f"{DATASET_ID}.{name}"

    def _load_data(self, **_: Any) -> None:
        """Load test data into a BigQuery instance."""

        credentials, default_project_id = google.auth.default(
            scopes=EXTERNAL_DATA_SCOPES
        )

        project_id = (
            os.environ.get(PROJECT_ID_ENV_VAR, default_project_id) or DEFAULT_PROJECT_ID
        )

        client = bq.Client(
            project=project_id,
            credentials=credentials,
            default_query_job_config=bq.QueryJobConfig(use_query_cache=False),
        )
        assert client.default_query_job_config.use_query_cache is False

        try:
            client.query("SELECT 1")
        except gexc.Forbidden:
            pytest.skip("User does not have permission to create dataset")

        query = client.query(f"CREATE SCHEMA IF NOT EXISTS {DATASET_ID}")
        query.result()

        path = self.script_dir.joinpath(f"{self.name()}.sql")
        ddl = path.read_text().format(dataset=DATASET_ID)
        query = client.query(ddl)
        query.result()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> Backend:  # noqa: ARG004
        """Connect to the test project and dataset."""
        credentials, default_project_id = google.auth.default(
            scopes=EXTERNAL_DATA_SCOPES
        )

        project_id = (
            os.environ.get(PROJECT_ID_ENV_VAR, default_project_id) or DEFAULT_PROJECT_ID
        )
        con = ibis.bigquery.connect(
            project_id=project_id, dataset_id=DATASET_ID, credentials=credentials, **kw
        )
        # disable the query cache to avoid invalid results
        #
        # it's rare, but it happens
        # https://github.com/googleapis/python-bigquery/issues/1845
        con.client.default_query_job_config.use_query_cache = False
        expr = ibis.literal(1)
        try:
            con.execute(expr)
        except gexc.Forbidden:
            pytest.skip(
                f"User does not have access to execute queries against BigQuery project: {project_id}"
            )
        else:
            return con
