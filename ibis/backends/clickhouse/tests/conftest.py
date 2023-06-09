from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Callable

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import (
    RoundHalfToEven,
    ServiceBackendTest,
    ServiceSpec,
    UnorderedComparator,
)

if TYPE_CHECKING:
    from pathlib import Path

CLICKHOUSE_HOST = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
CLICKHOUSE_PORT = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 8123))
CLICKHOUSE_USER = os.environ.get('IBIS_TEST_CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASS = os.environ.get('IBIS_TEST_CLICKHOUSE_PASSWORD', '')
IBIS_TEST_CLICKHOUSE_DB = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')


class TestConf(UnorderedComparator, ServiceBackendTest, RoundHalfToEven):
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supported_to_timestamp_units = {'s'}
    supports_floating_modulus = False
    supports_json = False

    @property
    def native_bool(self) -> bool:
        [(value,)] = self.connection.con.query("SELECT true").result_set
        return isinstance(value, bool)

    @classmethod
    def service_spec(cls, data_dir: Path) -> ServiceSpec:
        return ServiceSpec(
            name=cls.name(),
            data_volume="/var/lib/clickhouse/user_files/ibis",
            files=data_dir.joinpath("parquet").glob("*.parquet"),
        )

    @staticmethod
    def _load_data(
        data_dir: Path,
        script_dir: Path,
        host: str = CLICKHOUSE_HOST,
        port: int = CLICKHOUSE_PORT,
        user: str = CLICKHOUSE_USER,
        password: str = CLICKHOUSE_PASS,
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
        cc = pytest.importorskip("clickhouse_connect")

        client = cc.get_client(
            host=host,
            port=port,
            user=user,
            password=password,
            settings={
                "allow_experimental_object_type": 1,
                "output_format_json_named_tuples_as_objects": 1,
            },
        )

        with contextlib.suppress(cc.driver.exceptions.DatabaseError):
            client.command(f"CREATE DATABASE {database} ENGINE = Atomic")

        with open(script_dir / 'schema' / 'clickhouse.sql') as schema:
            for stmt in filter(None, map(str.strip, schema.read().split(";"))):
                client.command(stmt)

    @staticmethod
    def connect(data_directory: Path):
        pytest.importorskip("clickhouse_connect")
        return ibis.clickhouse.connect(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            password=CLICKHOUSE_PASS,
            database=IBIS_TEST_CLICKHOUSE_DB,
            user=CLICKHOUSE_USER,
        )

    @staticmethod
    def greatest(f: Callable[..., ir.Value], *args: ir.Value) -> ir.Value:
        if len(args) > 2:
            raise NotImplementedError(
                'Clickhouse does not support more than 2 arguments to greatest'
            )
        return f(*args)

    @staticmethod
    def least(f: Callable[..., ir.Value], *args: ir.Value) -> ir.Value:
        if len(args) > 2:
            raise NotImplementedError(
                'Clickhouse does not support more than 2 arguments to least'
            )
        return f(*args)


@pytest.fixture(scope='module')
def con(tmp_path_factory, data_directory, script_directory, worker_id):
    return TestConf.load_data(
        data_directory,
        script_directory,
        tmp_path_factory,
        worker_id,
    ).connect(data_directory)


@pytest.fixture(scope='module')
def db(con):
    return con.database()


@pytest.fixture(scope='module')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='module')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture
def translate():
    from ibis.backends.clickhouse.compiler.values import translate_val

    def t(*args, **kwargs):
        cache = kwargs.pop("cache", {})
        # we don't care about table aliases for the purposes of testing
        # individual function calls/expressions
        res = translate_val(*args, aliases={}, cache=cache, **kwargs)
        try:
            return res.sql(dialect="clickhouse")
        except AttributeError:
            return res

    return t
