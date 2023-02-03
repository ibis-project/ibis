from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundHalfToEven, UnorderedComparator

CLICKHOUSE_HOST = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
CLICKHOUSE_PORT = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 9000))
CLICKHOUSE_USER = os.environ.get('IBIS_TEST_CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASS = os.environ.get('IBIS_TEST_CLICKHOUSE_PASSWORD', '')
IBIS_TEST_CLICKHOUSE_DB = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')


class TestConf(UnorderedComparator, BackendTest, RoundHalfToEven):
    check_dtype = False
    supports_window_operations = False
    returned_timestamp_unit = 's'
    supported_to_timestamp_units = {'s'}
    supports_floating_modulus = False
    supports_json = False

    @property
    def native_bool(self) -> bool:
        [(value,)] = self.connection._client.execute("SELECT true")
        return isinstance(value, bool)

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
        clickhouse_driver = pytest.importorskip("clickhouse_driver")

        client = clickhouse_driver.Client(
            host=host, port=port, user=user, password=password
        )

        client.execute(f"DROP DATABASE IF EXISTS {database}")
        client.execute(f"CREATE DATABASE {database} ENGINE = Atomic")

        client.execute("DROP DATABASE IF EXISTS tmptables")
        client.execute("CREATE DATABASE tmptables ENGINE = Atomic")

        client.execute(f"USE {database}")
        client.execute("SET allow_experimental_object_type = 1")
        client.execute("SET output_format_json_named_tuples_as_objects = 1")

        with open(script_dir / 'schema' / 'clickhouse.sql') as schema:
            for stmt in filter(None, map(str.strip, schema.read().split(";"))):
                client.execute(stmt)

    @staticmethod
    def connect(data_directory: Path):
        pytest.importorskip("clickhouse_driver")
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
