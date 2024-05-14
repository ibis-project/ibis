from __future__ import annotations

import pytest

from ibis.backends.pyspark.tests.conftest import TestConfForStreaming


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    backend_test = TestConfForStreaming.load_data(data_dir, tmp_path_factory, worker_id)
    return backend_test.connection


@pytest.fixture
def session(con):
    return con._session
