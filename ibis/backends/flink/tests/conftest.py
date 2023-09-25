from __future__ import annotations

from typing import Any

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    supports_structs = False
    deps = "pandas", "pyflink"

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        """Flink backend is created in batch mode by default. This is to
        comply with the assumption that the tests under ibis/ibis/backends/tests/
        are for batch (storage or processing) backends.
        """
        from pyflink.table import EnvironmentSettings, TableEnvironment

        env_settings = EnvironmentSettings.in_batch_mode()
        table_env = TableEnvironment.create(env_settings)
        return ibis.flink.connect(table_env, **kw)

    def _load_data(self, **_: Any) -> None:
        import pandas as pd

        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            self.connection.create_table(table_name, pd.read_parquet(path))


class TestConfForStreaming(TestConf):
    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        """Flink backend is created in streaming mode here. To be used
        in the tests under ibis/ibis/backends/flink/tests/.
        """
        from pyflink.table import EnvironmentSettings, TableEnvironment

        env_settings = EnvironmentSettings.in_streaming_mode()
        table_env = TableEnvironment.create(env_settings)
        return ibis.flink.connect(table_env, **kw)


@pytest.fixture
def simple_schema():
    return [
        ("a", "int8"),
        ("b", "int16"),
        ("c", "int32"),
        ("d", "int64"),
        ("e", "float32"),
        ("f", "float64"),
        ("g", "string"),
        ("h", "boolean"),
        ("i", "timestamp"),
        ("j", "date"),
        ("k", "time"),
    ]


@pytest.fixture
def simple_table(simple_schema):
    return ibis.table(simple_schema, name="table")


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    return TestConfForStreaming.load_data(
        data_dir, tmp_path_factory, worker_id
    ).connection


@pytest.fixture(scope="session")
def db(con):
    return con.database()


@pytest.fixture(scope="session")
def alltypes(con):
    return con.tables.functional_alltypes
