from __future__ import absolute_import

import os

import pytest

from ibis.tests.all.config.backendtestconfiguration import (
    BackendTestConfiguration,
    UnorderedSeriesComparator,
)


class BigQuery(UnorderedSeriesComparator, BackendTestConfiguration):
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
