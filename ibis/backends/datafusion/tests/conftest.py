from __future__ import annotations

from typing import Any

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    # check_names = False
    # supports_divide_by_zero = True
    # returned_timestamp_unit = 'ns'
    supports_structs = False
    supports_json = False
    supports_arrays = False
    stateful = False
    deps = ("datafusion",)

    def _load_data(self, **_: Any) -> None:
        con = self.connection
        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            con.register(path, table_name=table_name)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.datafusion.connect(**kw)


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()
