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
        return ibis.sqlite.connect(str(path))  # type: ignore

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        t = super().functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast('timestamp'))


@pytest.fixture(scope="session")
def dbpath(data_directory):
    default = str(data_directory / 'ibis_testing.db')
    return os.environ.get('IBIS_TEST_SQLITE_DATABASE', default)


@pytest.fixture(scope="session")
def con(dbpath):
    return ibis.sqlite.connect(dbpath)


@pytest.fixture(scope="session")
def dialect():
    import sqlalchemy as sa

    return sa.dialects.sqlite.dialect()


@pytest.fixture(scope="session")
def translate(dialect):
    from ibis.backends.sqlite import Backend

    context = Backend.compiler.make_context()
    return lambda expr: str(
        Backend.compiler.translator_class(expr, context)
        .get_result()
        .compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture(scope="session")
def sqla_compile(dialect):
    return lambda expr: str(
        expr.compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_sqla(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()
