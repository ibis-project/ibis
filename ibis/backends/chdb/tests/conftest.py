from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


IBIS_TEST_CHDB_DB = os.environ.get("IBIS_TEST_DATA_DB", "ibis_testing")


class TestConf(BackendTest):
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = "s"
    supported_to_timestamp_units = {"s"}
    supports_floating_modulus = False
    supports_json = False
    force_sort = True
    rounding_method = "half_to_even"
    deps = ("chdb",)

    @property
    def native_bool(self) -> bool:
        return True

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("parquet").glob("*.parquet")

    @property
    def ddl_script(self) -> Iterable[str]:
        parquet_dir = self.data_dir / "parquet"
        for sql in super().ddl_script:
            yield sql.format(parquet_dir=parquet_dir)

    def _load_data(self, *, database: str = IBIS_TEST_CHDB_DB, **_) -> None:
        """Load test data into a ClickHouse backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        con = self.connection.con

        con.query(f"CREATE DATABASE {database} ENGINE = Atomic")
        con.query(f"USE {database}")
        for sql in self.ddl_script:
            sql = sql.replace("\n", " ")
            con.query(sql)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        return ibis.chdb.connect()

    @staticmethod
    def greatest(f: Callable[..., ir.Value], *args: ir.Value) -> ir.Value:
        if len(args) > 2:
            raise NotImplementedError(
                "Clickhouse does not support more than 2 arguments to greatest"
            )
        return f(*args)

    @staticmethod
    def least(f: Callable[..., ir.Value], *args: ir.Value) -> ir.Value:
        if len(args) > 2:
            raise NotImplementedError(
                "Clickhouse does not support more than 2 arguments to least"
            )
        return f(*args)


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="session")
def alltypes(con):
    return con.tables.functional_alltypes


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()
