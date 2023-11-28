from __future__ import annotations

from typing import Any

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest


class TestConf(BackendTest):
    supports_structs = False
    force_sort = True
    deps = "pandas", "pyflink"

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        """Flink backend is created in batch mode by default. This is to
        comply with the assumption that the tests under ibis/ibis/backends/tests/
        are for batch (storage or processing) backends.
        """
        import os

        from pyflink.java_gateway import get_gateway
        from pyflink.table import StreamTableEnvironment
        from pyflink.table.table_environment import StreamExecutionEnvironment

        # connect with Flink remote cluster to run the unit tests
        gateway = get_gateway()
        string_class = gateway.jvm.String
        string_array = gateway.new_array(string_class, 0)
        env_settings = (
            gateway.jvm.org.apache.flink.table.api.EnvironmentSettings.inBatchMode()
        )
        stream_env = gateway.jvm.org.apache.flink.streaming.api.environment.StreamExecutionEnvironment
        flink_cluster_addr = os.environ.get("FLINK_REMOTE_CLUSTER_ADDR", "localhost")
        flink_cluster_port = int(os.environ.get("FLINK_REMOTE_CLUSTER_PORT", "8081"))
        j_stream_exection_environment = stream_env.createRemoteEnvironment(
            flink_cluster_addr,
            flink_cluster_port,
            env_settings.getConfiguration(),
            string_array,
        )

        env = StreamExecutionEnvironment(j_stream_exection_environment)
        stream_table_env = StreamTableEnvironment.create(env)
        return ibis.flink.connect(stream_table_env, **kw)

    def _load_data(self, **_: Any) -> None:
        import pandas as pd

        from ibis.backends.tests.data import json_types

        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            self.connection.create_table(table_name, pd.read_parquet(path))

        self.connection.create_table("json_t", json_types)


class TestConfForStreaming(TestConf):
    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        """Flink backend is created in streaming mode here. To be used
        in the tests under ibis/ibis/backends/flink/tests/.
        We only use mini cluster here for simplicity.
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
