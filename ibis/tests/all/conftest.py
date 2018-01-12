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
def backend_con(backend):
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
def backend_alltypes(backend, backend_con):
    return backend.functional_alltypes(backend_con)


@pytest.fixture(scope='session')
def analytic_alltypes(backend_alltypes):
    return backend_alltypes.groupby('string_col').order_by('id')


@pytest.fixture(scope='session')
def backend_df(backend_alltypes):
    return backend_alltypes.execute()
