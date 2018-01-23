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
    except AttributeError as e:
        pytest.skip(
            'Unable to import the {} backend: {}'.format(
                backend_module_name, e
            )
        )
    return backend.connect(module)


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
def valid_operations(registry, rewrites, backend):
    return (
        frozenset(registry) | frozenset(rewrites)
    ) - backend.additional_skipped_operations


@pytest.fixture(scope='session')
def alltypes(backend, con):
    return backend.functional_alltypes(con)


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes.groupby('string_col').order_by('id')


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()
