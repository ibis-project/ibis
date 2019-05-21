import os

import pytest

import ibis


@pytest.fixture(scope='module')
def dbpath(data_directory):
    default = str(data_directory / 'ibis_testing.db')
    path = os.environ.get('IBIS_TEST_SQLITE_DATABASE', default)
    if not os.path.exists(path):
        pytest.skip('SQLite testing db {} does not exist'.format(path))
    else:
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


@pytest.fixture
def dialect():
    import sqlalchemy as sa
    return sa.dialects.sqlite.dialect()


@pytest.fixture
def translate(dialect):
    from ibis.sql.sqlite.compiler import SQLiteDialect
    ibis_dialect = SQLiteDialect()
    context = ibis_dialect.make_context()
    return lambda expr: str(
        ibis_dialect.translator(expr, context).get_result().compile(
            dialect=dialect,
            compile_kwargs=dict(literal_binds=True)
        )
    )


@pytest.fixture
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
