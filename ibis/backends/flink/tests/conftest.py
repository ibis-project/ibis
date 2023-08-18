from __future__ import annotations

from typing import Any

import pytest

import ibis
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    supports_structs = False
    deps = ("pyflink",)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        from pyflink.table import EnvironmentSettings, TableEnvironment

        env_settings = EnvironmentSettings.in_batch_mode()
        table_env = TableEnvironment.create(env_settings)
        return ibis.flink.connect(table_env, **kw)

    def _load_data(self, **_: Any) -> None:
        for stmt in self.ddl_script:
            self.connection._t_env.execute_sql(stmt.format(data_dir=self.data_dir))


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
