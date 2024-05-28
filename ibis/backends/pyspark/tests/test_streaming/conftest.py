from __future__ import annotations

from unittest import mock

import pytest

from ibis.backends.pyspark import Backend
from ibis.backends.pyspark.tests.conftest import TestConfForStreaming


@pytest.fixture(scope="session", autouse=True)
def default_session_fixture():
    with mock.patch.object(Backend, "write_to_memory", write_to_memory, create=True):
        yield


def write_to_memory(self, expr, table_name):
    if self.mode == "batch":
        raise NotImplementedError
    df = self._session.sql(expr.compile())
    df.writeStream.format("memory").queryName(table_name).start()


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    backend_test = TestConfForStreaming.load_data(data_dir, tmp_path_factory, worker_id)
    return backend_test.connection


@pytest.fixture
def session(con):
    return con._session
