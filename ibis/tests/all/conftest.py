import pytest

import ibis

from ibis.tests.all import backends


@pytest.fixture(
    params=backends,
    scope='session',
    ids=lambda cls: cls.__name__.lower(),
)
def backend(request):
    return request.param


@pytest.fixture(scope='session')
def con(backend):
    backend_module_name = backend.__name__.lower()
    try:
        module = getattr(ibis, backend_module_name)
    except AttributeError:
        pytest.skip(
            'Unable to import the {} backend'.format(backend_module_name)
        )
    else:
        return backend.connect(module)


@pytest.fixture
def assertion_function(backend):
    return backend.assert_series_equal


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
