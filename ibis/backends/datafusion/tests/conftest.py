from __future__ import annotations

from typing import Any

import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.backends.tests.data import array_types, topk, win


class TestConf(BackendTest):
    # check_names = False
    # returned_timestamp_unit = 'ns'
    supports_structs = False
    supports_json = False
    supports_arrays = True
    supports_tpch = True
    stateful = False
    deps = ("datafusion",)
    # Query 1 seems to require a bit more room here
    tpch_absolute_tolerance = 0.11

    def _load_data(self, **_: Any) -> None:
        con = self.connection
        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            with pytest.warns(FutureWarning, match="v9.1"):
                con.register(path, table_name=table_name)
        # TODO: remove warnings and replace register when implementing 8858
        with pytest.warns(FutureWarning, match="v9.1"):
            con.register(array_types, table_name="array_types")
            con.register(win, table_name="win")
            con.register(topk, table_name="topk")

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.datafusion.connect(**kw)

    def load_tpch(self) -> None:
        con = self.connection
        for path in self.data_dir.joinpath("tpch", "sf=0.17", "parquet").glob(
            "*.parquet"
        ):
            table_name = path.with_suffix("").name
            con.read_parquet(path, table_name=table_name)


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()
