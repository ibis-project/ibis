from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.chdb import Session
from ibis.backends.tests.base import (
    BackendTest,
    RoundHalfToEven,
    UnorderedComparator,
)

if TYPE_CHECKING:
    from ibis.backends.base import BaseBackend

DEFAULT_DATABASE = os.environ.get("IBIS_TEST_DATA_DB", "ibis_testing")


class TestConf(UnorderedComparator, BackendTest, RoundHalfToEven):
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = "s"
    supported_to_timestamp_units = {"s"}
    supports_floating_modulus = False
    supports_json = False
    deps = ("chdb",)

    @property
    def native_bool(self) -> bool:
        [(value,)] = self.connection.con.query("SELECT true").result_set
        return isinstance(value, bool)

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

    def _load_data(self, *, database: str = DEFAULT_DATABASE, **_) -> None:
        con = Session()
        con.query(f"CREATE DATABASE {database} ENGINE = Atomic")

        for query in self.ddl_script:
            query = query.format(parquet_dir=self.data_dir / "parquet")
            con.query(query)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw) -> BaseBackend:
        # extension directory per test worker to prevent simultaneous downloads
        return ibis.chdb.connect(**kw)


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection
