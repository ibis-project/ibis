import os
import pytest

from ibis.compat import Path
from ibis.tests.backends import (Csv, Parquet, Pandas,
                                 SQLite, Postgres, MySQL,
                                 Clickhouse, Impala, BigQuery)


pytestmark = pytest.mark.backend


@pytest.fixture(scope='session')
def data_directory():
    root = Path(__file__).absolute().parents[3]

    default = root / 'testing' / 'ibis-testing-data'
    datadir = os.environ.get('IBIS_TEST_DATA_DIRECTORY', default)
    datadir = Path(datadir)

    if not datadir.exists():
        pytest.skip('test data directory not found')

    return datadir


@pytest.fixture(params=[
    pytest.param(Csv, marks=pytest.mark.csv),
    pytest.param(Parquet, marks=pytest.mark.parquet),
    pytest.param(Pandas, marks=pytest.mark.pandas),
    pytest.param(SQLite, marks=pytest.mark.sqlite),
    pytest.param(Postgres, marks=pytest.mark.postgres),
    pytest.param(MySQL, marks=pytest.mark.mysql),
    pytest.param(Clickhouse, marks=pytest.mark.clickhouse),
    pytest.param(BigQuery, marks=pytest.mark.bigquery),
    pytest.param(Impala, marks=pytest.mark.impala)
], scope='session')
def backend(request, data_directory):
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    return backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes()


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes.groupby('string_col').order_by('id')


@pytest.fixture(scope='session')
def df(backend):
    return backend.functional_alltypes_df()


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
