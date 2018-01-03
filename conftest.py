import os
import pytest

from ibis.compat import Path


@pytest.fixture(scope='session')
def data_directory():
    current = Path(__file__).absolute()
    default = current.parent / 'testing' / 'ibis-testing-data'
    datadir = os.environ.get('IBIS_TEST_DATA_DIRECTORY', default)
    datadir = Path(datadir)

    if not datadir.exists():
        pytest.skip('test data directory not found')

    return datadir


@pytest.fixture(scope='session')
def backend(data_directory):
    """Backend fixture should be overriden"""
    raise NotImplementedError


@pytest.fixture(scope='session')
def con(backend):
    raise NotImplementedError


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes()


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes.groupby('string_col').order_by('id')


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


# @pytest.fixture(scope='session')
# def dialect(con):
#     return con.dialect


# @pytest.fixture(scope='session')
# def translator(dialect):
#     return dialect.translator


# @pytest.fixture(scope='session')
# def registry(translator):
#     return translator._registry


# @pytest.fixture(scope='session')
# def rewrites(translator):
#     return translator._rewrites


# @pytest.fixture(scope='session')
# def valid_operations(registry, rewrites, backend):
#     return (
#         frozenset(registry) | frozenset(rewrites)
#     ) - backend.additional_skipped_operations
