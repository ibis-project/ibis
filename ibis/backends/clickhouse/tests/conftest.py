import os
from pathlib import Path
from typing import Callable

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import (
    BackendTest,
    RoundHalfToEven,
    UnorderedComparator,
)

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
    bool_is_int = True

    @staticmethod
    def connect(data_directory: Path):
        host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST', 'localhost')
        port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT', 9000))
        user = os.environ.get('IBIS_TEST_CLICKHOUSE_USER', 'default')
        password = os.environ.get('IBIS_TEST_CLICKHOUSE_PASSWORD', '')
        database = os.environ.get(
            'IBIS_TEST_CLICKHOUSE_DATABASE', 'ibis_testing'
        )
        return ibis.clickhouse.connect(
            host=host,
            port=port,
            password=password,
            database=database,
            user=user,
        )

    @staticmethod
    def greatest(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        if len(args) > 2:
            raise NotImplementedError(
                'Clickhouse does not support more than 2 arguments to greatest'
            )
        return f(*args)

    @staticmethod
    def least(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        if len(args) > 2:
            raise NotImplementedError(
                'Clickhouse does not support more than 2 arguments to least'
            )
        return f(*args)


@pytest.fixture(scope='module')
def con():
    return ibis.clickhouse.connect(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASS,
        database=IBIS_TEST_CLICKHOUSE_DB,
    )


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
    from ibis.backends.clickhouse.compiler import (
        ClickhouseCompiler,
        ClickhouseExprTranslator,
    )

    context = ClickhouseCompiler.make_context()
    return lambda expr: ClickhouseExprTranslator(expr, context).get_result()
