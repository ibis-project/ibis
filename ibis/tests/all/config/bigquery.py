from __future__ import absolute_import

import os

import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration,
    UnorderedSeriesComparator,
)


class BigQuery(UnorderedSeriesComparator, BackendTestConfiguration):

    required_modules = 'google.cloud.bigquery', 'google.auth'

    @classmethod
    def connect(cls, backend):
        import google.auth

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
        except google.auth.exceptions.DefaultCredentialsError:
            pytest.skip('No Google/BigQuery credentials found')
