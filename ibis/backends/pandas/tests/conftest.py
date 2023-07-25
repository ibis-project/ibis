from __future__ import annotations

from typing import Any

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundHalfToEven
from ibis.backends.tests.data import array_types, json_types, struct_types, win


class TestConf(BackendTest, RoundHalfToEven):
    check_names = False
    supported_to_timestamp_units = BackendTest.supported_to_timestamp_units | {"ns"}
    supports_divide_by_zero = True
    returned_timestamp_unit = "ns"
    stateful = False
    deps = ("pandas",)

    def _load_data(self, **_: Any) -> None:
        import pandas as pd

        con = self.connection
        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            con.create_table(table_name, pd.read_parquet(path))
        con.create_table("array_types", array_types, overwrite=True)
        con.create_table("struct", struct_types, overwrite=True)
        con.create_table("win", win, overwrite=True)
        con.create_table("json_t", json_types, overwrite=True)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.pandas.connect(**kw)
