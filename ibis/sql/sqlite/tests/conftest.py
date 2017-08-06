import os

import pytest

import ibis

sa = pytest.importorskip('sqlalchemy')

from ibis.sql.sqlite.compiler import SQLiteExprTranslator  # noqa: E402


@pytest.fixture(scope='module')
def dbpath():
    # If we haven't defined an environment variable with the path of the SQLite
    # database, assume it's in $PWD
    path = os.environ.get('IBIS_TEST_SQLITE_DB_PATH', 'ibis_testing.db')
    if not os.path.exists(path):
        pytest.skip("sql testing db does not exist!")
    return path


@pytest.yield_fixture(scope='module')
def con(dbpath):
    con = ibis.sqlite.connect(dbpath)
    try:
        yield con
    finally:
        con.con.dispose()


@pytest.fixture(scope='module')
def db(con):
    return con.database()


@pytest.fixture(scope='module')
def dialect():
    return sa.dialects.sqlite.dialect()


@pytest.fixture(scope='module')
def translate(dialect):
    return lambda expr: str(
        SQLiteExprTranslator(expr).get_result().compile(
            dialect=dialect,
            compile_kwargs=dict(literal_binds=True)
        )
    )


@pytest.fixture(scope='module')
def sqla_compile(dialect):
    return lambda expr: str(
        expr.compile(dialect=dialect, compile_kwargs=dict(literal_binds=True))
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
