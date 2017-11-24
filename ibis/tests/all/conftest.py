import collections
import getpass
import os

import ibis

import pytest


backend_connect_functions = collections.OrderedDict()


def test_connect(backend):
    def wrapper(connect_function):
        backend_connect_functions[backend] = connect_function
        return connect_function
    return wrapper


@test_connect('impala')
def connect_impala(backend):
    pass


@test_connect('sqlite')
def connect_sqlite(backend):
    path = os.environ.get('IBIS_TEST_SQLITE_DB_PATH', 'ibis_testing.db')
    if not os.path.exists(path):
        pytest.skip('SQLite testing db {} does not exist'.format(path))
    else:
        con = backend.connect(path)
        try:
            yield con
        finally:
            con.con.dispose()


@test_connect('postgres')
def connect_postgres(backend):
    PG_USER = os.environ.get(
        'IBIS_POSTGRES_USER',
        os.environ.get('PGUSER', getpass.getuser())
    )
    PG_PASS = os.environ.get('IBIS_POSTGRES_PASS', os.environ.get('PGPASS'))
    PG_HOST = os.environ.get('PGHOST', 'localhost')
    IBIS_TEST_POSTGRES_DB = os.environ.get(
        'IBIS_TEST_POSTGRES_DB',
        os.environ.get('PGDATABASE', 'ibis_testing')
    )
    return backend.connect(
        host=PG_HOST,
        user=PG_USER,
        password=PG_PASS,
        database=IBIS_TEST_POSTGRES_DB,
    )


@test_connect('clickhouse')
def connect_clickhouse(backend):
    pass


@test_connect('bigquery')
def connect_bigquery(backend):
    ga = pytest.importorskip('google.auth')

    PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID')
    DATASET_ID = 'testing'

    try:
        return backend.connect(PROJECT_ID, DATASET_ID)
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip("no credentials found, skipping")


@test_connect('pandas')
def connect_pandas(backend):
    pytest.importorskip('multipledispatch')
    pytest.skip()


@test_connect('csv')
def connect_csv(backend):
    pytest.skip()


@test_connect('hdf5')
def connect_hdf5(backend):
    pytest.skip()


@test_connect('parquet')
def connect_parquet(backend):
    pytest.skip()


@pytest.fixture(
    params=list(backend_connect_functions.items()), scope='session'
)
def con(request):
    backend, connect_function = request.param
    try:
        module = getattr(ibis, backend)
    except AttributeError:
        pytest.skip('could not connect to the {} backend'.format(backend))
    else:
        return connect_function(module)


@pytest.fixture(scope='session')
def db(con):
    return con.database()


@pytest.fixture(scope='session')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()
