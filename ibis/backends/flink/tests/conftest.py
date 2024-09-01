from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.backends.tests.data import array_types, json_types, struct_types, topk, win

if TYPE_CHECKING:
    from pyflink.table import StreamTableEnvironment

TEST_TABLES["functional_alltypes"] = ibis.schema(
    {
        "id": "int32",
        "bool_col": "boolean",
        "tinyint_col": "int8",
        "smallint_col": "int16",
        "int_col": "int32",
        "bigint_col": "int64",
        "float_col": "float32",
        "double_col": "float64",
        "date_string_col": "string",
        "string_col": "string",
        "timestamp_col": "timestamp(3)",  # overriding the higher level fixture with precision because Flink's
        # watermark must use a field of type TIMESTAMP(p) or TIMESTAMP_LTZ(p), where 'p' is from 0 to 3
        "year": "int32",
        "month": "int32",
    }
)


def get_table_env(
    local_env: bool,
    streaming_mode: bool,
) -> StreamTableEnvironment:
    if local_env:
        from pyflink.table import EnvironmentSettings, TableEnvironment

        env_settings = (
            EnvironmentSettings.in_streaming_mode()
            if streaming_mode
            else EnvironmentSettings.in_batch_mode()
        )
        table_env = TableEnvironment.create(env_settings)

    else:
        import os

        from pyflink.java_gateway import get_gateway
        from pyflink.table import StreamTableEnvironment
        from pyflink.table.table_environment import StreamExecutionEnvironment

        # Connect with Flink remote cluster to run the unit tests
        gateway = get_gateway()
        string_class = gateway.jvm.String
        string_array = gateway.new_array(string_class, 0)
        env_settings = (
            gateway.jvm.org.apache.flink.table.api.EnvironmentSettings.inStreamingMode()
            if streaming_mode
            else gateway.jvm.org.apache.flink.table.api.EnvironmentSettings.inBatchMode()
        )
        stream_env = gateway.jvm.org.apache.flink.streaming.api.environment.StreamExecutionEnvironment
        flink_cluster_addr = os.environ.get("FLINK_REMOTE_CLUSTER_ADDR", "localhost")
        flink_cluster_port = int(os.environ.get("FLINK_REMOTE_CLUSTER_PORT", "8081"))
        j_execution_environment = stream_env.createRemoteEnvironment(
            flink_cluster_addr,
            flink_cluster_port,
            env_settings.getConfiguration(),
            string_array,
        )

        env = StreamExecutionEnvironment(j_execution_environment)
        table_env = StreamTableEnvironment.create(env)

    table_config = table_env.get_config()
    table_config.set("table.local-time-zone", "UTC")

    return table_env


class TestConf(BackendTest):
    force_sort = True
    stateful = False
    supports_map = True
    deps = "pandas", "pyflink"

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        """Flink backend is created in batch mode by default. This is to
        comply with the assumption that the tests under ibis/ibis/backends/tests/
        are for batch (storage or processing) backends.
        """

        table_env = get_table_env(local_env=False, streaming_mode=False)
        return ibis.flink.connect(table_env, **kw)

    def _load_data(self, **_: Any) -> None:
        con = self.connection

        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            con.create_table(table_name, pd.read_parquet(path), temp=True)

        con.create_table("array_types", array_types, temp=True)
        con.create_table("json_t", json_types, temp=True)
        con.create_table("struct", struct_types, temp=True)
        con.create_table("win", win, temp=True)
        con.create_table(
            "map",
            pd.DataFrame(
                {
                    "idx": [1, 2],
                    "kv": [{"a": 1, "b": 2, "c": 3}, {"d": 4, "e": 5, "f": 6}],
                }
            ),
            schema=ibis.schema({"idx": "int64", "kv": "map<string, int64>"}),
            temp=True,
        )
        con.create_table("topk", topk, temp=True)


class TestConfForStreaming(TestConf):
    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        """Flink backend is created in streaming mode here. To be used
        in the tests under ibis/ibis/backends/flink/tests/.
        We only use mini cluster here for simplicity.
        """

        table_env = get_table_env(local_env=True, streaming_mode=True)
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


@pytest.fixture
def awards_players_schema():
    return TEST_TABLES["awards_players"]


@pytest.fixture
def functional_alltypes_schema():
    return TEST_TABLES["functional_alltypes"]


@pytest.fixture
def csv_source_configs():
    def generate_csv_configs(csv_file):
        return {
            "connector": "filesystem",
            "path": f"ci/ibis-testing-data/csv/{csv_file}.csv",
            "format": "csv",
            "csv.ignore-parse-errors": "true",
        }

    return generate_csv_configs


@pytest.fixture(scope="session")
def functional_alltypes_no_header(tmpdir_factory, data_dir):
    file = tmpdir_factory.mktemp("data") / "functional_alltypes.csv"
    with (
        open(data_dir / "csv" / "functional_alltypes.csv") as reader,
        open(str(file), mode="w") as writer,
    ):
        reader.readline()  # read the first line and discard it
        for line in reader:
            writer.write(line)
    return file


@pytest.fixture(scope="session", autouse=True)
def functional_alltypes_with_watermark(con, functional_alltypes_no_header):
    # create a streaming table with watermark for testing event-time based ops
    t = con.create_table(
        "functional_alltypes_with_watermark",
        schema=TEST_TABLES["functional_alltypes"],
        tbl_properties={
            "connector": "filesystem",
            "path": functional_alltypes_no_header,
            "format": "csv",
        },
        watermark=ibis.watermark("timestamp_col", ibis.interval(seconds=10)),
        temp=True,
    )
    return t
