import os

import pytest

from ibis.tests.all.config.backendtestconfiguration import BackendTestConfiguration


class BigQuery(BackendTestConfiguration):
    @classmethod
    def connect(cls, backend):
        ga = pytest.importorskip('google.auth')

        project_id = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID')

        if project_id is None:
            pytest.skip(
                'Environment variable GOOGLE_BIGQUERY_PROJECT_ID not defined'
            )

        if not project_id:
            pytest.skip(
                'Environment variable GOOGLE_BIGQUERY_PROJECT_ID is empty'
            )

        dataset_id = 'testing'

        try:
            return backend.connect(project_id, dataset_id)
        except ga.exceptions.DefaultCredentialsError:
            pytest.skip('no credentials found, skipping')

    @classmethod
    def assert_series_equal(cls, left, right, *args, **kwargs):
        return super(BigQuery, cls).assert_series_equal(
            left.value_counts().sort_index(),
            right.value_counts().sort_index(),
            *args,
            **kwargs
        )
