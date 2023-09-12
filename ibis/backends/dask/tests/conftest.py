from __future__ import annotations

from typing import Any

import dask
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.pandas.tests.conftest import TestConf as PandasTest
from ibis.backends.tests.data import array_types, json_types, win

# FIXME Dask issue with non deterministic groupby results, relates to the
# shuffle method on a local cluster. Manually setting the shuffle method
# avoids the issue https://github.com/dask/dask/issues/10034.
dask.config.set({"dataframe.shuffle.method": "tasks"})

# TODO: support pyarrow string column types across ibis
dask.config.set({"dataframe.convert-string": False})

# It's necessary that NPARTITIONS > 1 in order to test cross partitioning bugs.
NPARTITIONS = 2


@pytest.fixture(scope="module")
def npartitions():
    return NPARTITIONS


class TestConf(PandasTest):
    supports_structs = False
    deps = ("dask.dataframe",)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.dask.connect(**kw)

    def _load_data(self, **_: Any) -> None:
        import dask.dataframe as dd

        con = self.connection
        for table_name in TEST_TABLES:
            path = self.data_dir / "parquet" / f"{table_name}.parquet"
            con.create_table(
                table_name,
                dd.from_pandas(pd.read_parquet(path), npartitions=NPARTITIONS),
            )

        con.create_table(
            "array_types",
            dd.from_pandas(array_types, npartitions=NPARTITIONS),
            overwrite=True,
        )
        con.create_table(
            "win", dd.from_pandas(win, npartitions=NPARTITIONS), overwrite=True
        )
        con.create_table(
            "json_t",
            dd.from_pandas(json_types, npartitions=NPARTITIONS),
            overwrite=True,
        )

    @classmethod
    def assert_series_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault("check_dtype", cls.check_dtype)
        kwargs.setdefault("check_names", cls.check_names)
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        tm.assert_series_equal(left, right, *args, **kwargs)


@pytest.fixture
def dataframe(npartitions):
    dd = pytest.importorskip("dask.dataframe")

    return dd.from_pandas(
        pd.DataFrame(
            {
                "plain_int64": list(range(1, 4)),
                "plain_strings": list("abc"),
                "dup_strings": list("dad"),
            }
        ),
        npartitions=npartitions,
    )


@pytest.fixture
def core_client(dataframe):
    return ibis.dask.connect({"df": dataframe})


@pytest.fixture
def ibis_table(core_client):
    return core_client.table("df")
