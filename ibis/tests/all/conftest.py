import collections
import getpass
import operator
import os

import pandas as pd
import ibis

import pytest


backend_connect_functions = collections.OrderedDict()


def test_connection(backend):
    def wrapper(connect_function):
        backend_connect_functions[backend] = connect_function
        return connect_function
    return wrapper


@test_connection('impala')
def connect_impala(backend):
    pytest.skip('Skipping {}'.format(backend.__name__))


@test_connection('sqlite')
def connect_sqlite(backend):
    path = os.environ.get('IBIS_TEST_SQLITE_DB_PATH', 'ibis_testing.db')
    if not os.path.exists(path):
        pytest.skip('SQLite testing db {} does not exist'.format(path))
    else:
        con = backend.connect(path)
        return con


@test_connection('postgres')
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


@test_connection('clickhouse')
def connect_clickhouse(backend):
    pytest.skip('Skipping {}'.format(backend.__name__))


@test_connection('bigquery')
def connect_bigquery(backend):
    ga = pytest.importorskip('google.auth')

    PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID')
    DATASET_ID = 'testing'

    try:
        return backend.connect(PROJECT_ID, DATASET_ID)
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip("no credentials found, skipping")


@test_connection('pandas')
def connect_pandas(backend):
    pytest.importorskip('multipledispatch')
    test_data_directory = os.environ.get('IBIS_TEST_DATA_DIRECTORY')
    filename = os.path.join(test_data_directory, 'functional_alltypes.csv')
    if not os.path.exists(filename):
        pytest.skip('test data set functional_alltypes not found')
    else:
        return backend.connect({
            'functional_alltypes': pd.read_csv(filename, index_col=None)
        })


@test_connection('csv')
def connect_csv(backend):
    test_data_directory = os.environ.get('IBIS_TEST_DATA_DIRECTORY')
    filename = os.path.join(test_data_directory, 'functional_alltypes.csv')
    if not os.path.exists(test_data_directory):
        pytest.skip('test data directory not found')
    if not os.path.exists(filename):
        pytest.skip(
            'test data set functional_alltypes.csv not found in '
            'test data directory'
        )
    else:
        return backend.connect(test_data_directory)


@test_connection('hdf5')
def connect_hdf5(backend):
    pytest.skip('Skipping {}'.format(backend.__name__))


@test_connection('parquet')
def connect_parquet(backend):
    test_data_directory = os.environ.get('IBIS_TEST_DATA_DIRECTORY')
    filename = os.path.join(test_data_directory, 'functional_alltypes.parquet')
    if not os.path.exists(test_data_directory):
        pytest.skip('test data directory not found')
    if not os.path.exists(filename):
        pytest.skip(
            'test data set functional_alltypes.parquet not found in '
            'test data directory'
        )
    else:
        return backend.connect(test_data_directory)


@pytest.fixture(
    params=list(backend_connect_functions.items()),
    scope='session',
    ids=operator.itemgetter(0),
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
def dialect(con):
    return con.dialect


@pytest.fixture(scope='session')
def translator(dialect):
    return dialect.translator


@pytest.fixture(scope='session')
def registry(translator):
    return translator._registry


@pytest.fixture(scope='session')
def rewrites(translator):
    return translator._rewrites


@pytest.fixture(scope='session')
def valid_operations(registry, rewrites):
    return frozenset(registry) | frozenset(rewrites)


@pytest.fixture(scope='session')
def db(con):
    return con.database()


@pytest.fixture(scope='session')
def alltypes(db):
    return db.functional_alltypes


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()
