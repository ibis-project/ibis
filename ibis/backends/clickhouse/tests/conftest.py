from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, Callable

import pytest

import ibis
import ibis.expr.types as ir
from ibis import util
from ibis.backends.tests.base import (
    RoundHalfToEven,
    ServiceBackendTest,
    UnorderedComparator,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

CLICKHOUSE_HOST = os.environ.get("IBIS_TEST_CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.environ.get("IBIS_TEST_CLICKHOUSE_PORT", 8123))
CLICKHOUSE_USER = os.environ.get("IBIS_TEST_CLICKHOUSE_USER", "default")
CLICKHOUSE_PASS = os.environ.get("IBIS_TEST_CLICKHOUSE_PASSWORD", "")
IBIS_TEST_CLICKHOUSE_DB = os.environ.get("IBIS_TEST_DATA_DB", "ibis_testing")


class TestConf(UnorderedComparator, ServiceBackendTest, RoundHalfToEven):
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = "s"
    supported_to_timestamp_units = {"s"}
    supports_floating_modulus = False
    supports_json = False
    data_volume = "/var/lib/clickhouse/user_files/ibis"
    service_name = "clickhouse"
    deps = ("clickhouse_connect",)

    @property
    def native_bool(self) -> bool:
        [(value,)] = self.connection.con.query("SELECT true").result_set
        return isinstance(value, bool)

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("parquet").glob("*.parquet")

    def _load_data(
        self,
        *,
        database: str = IBIS_TEST_CLICKHOUSE_DB,
        **_,
    ) -> None:
        """Load test data into a ClickHouse backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        import clickhouse_connect as cc

        con = self.connection
        client = con.con

        with contextlib.suppress(cc.driver.exceptions.DatabaseError):
            client.command(f"CREATE DATABASE {database} ENGINE = Atomic")

        util.consume(map(client.command, self.ddl_script))

    def postload(self, **kw: Any):
        # reconnect to set the database to the test database
        self.connection = self.connect(database=IBIS_TEST_CLICKHOUSE_DB, **kw)

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        return ibis.clickhouse.connect(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            password=CLICKHOUSE_PASS,
            user=CLICKHOUSE_USER,
            **kw,
        )

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
def db(con):
    return con.database()


@pytest.fixture(scope="session")
def alltypes(con):
    return con.tables.functional_alltypes


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()
