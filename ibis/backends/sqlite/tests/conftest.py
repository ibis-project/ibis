import os
from pathlib import Path

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    check_dtype = False
    returned_timestamp_unit = 's'

    @staticmethod
    def connect(data_directory: Path):
        path = Path(
            os.environ.get(
                'IBIS_TEST_SQLITE_DATABASE', data_directory / 'ibis_testing.db'
            )
        )
        return ibis.sqlite.connect(str(path))

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        t = self.db.functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast('timestamp'))


@pytest.fixture(scope='module')
def dbpath(data_directory):
    default = str(data_directory / 'ibis_testing.db')
    return os.environ.get('IBIS_TEST_SQLITE_DATABASE', default)


@pytest.fixture(scope='module')
def con(dbpath):
    return ibis.sqlite.connect(dbpath)


@pytest.fixture(scope='module')
def db(con):
    return con.database()


@pytest.fixture
def dialect():
    import sqlalchemy as sa

    return sa.dialects.sqlite.dialect()


@pytest.fixture
def translate(dialect):
    from ibis.backends.sqlite import SQLiteClient

    client = SQLiteClient
    context = client.compiler.make_context()
    return lambda expr: str(
        client.compiler.translator(expr, context)
        .get_result()
        .compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture
def sqla_compile(dialect):
    return lambda expr: str(
        expr.compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture(scope='module')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='module')
def alltypes_sqla(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture(scope='module')
def df(alltypes):
    return alltypes.execute()
