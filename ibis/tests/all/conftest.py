import sys
import pytest

from ibis.tests.backends import (Csv, Parquet, Pandas,
                                 SQLite, Postgres, MySQL, MapD,
                                 Clickhouse, Impala, BigQuery)


pytestmark = pytest.mark.backend

params_backend = [
    pytest.param(Csv, marks=pytest.mark.csv),
    pytest.param(Parquet, marks=pytest.mark.parquet),
    pytest.param(Pandas, marks=pytest.mark.pandas),
    pytest.param(SQLite, marks=pytest.mark.sqlite),
    pytest.param(Postgres, marks=pytest.mark.postgres),
    pytest.param(MySQL, marks=pytest.mark.mysql),
    pytest.param(Clickhouse, marks=pytest.mark.clickhouse),
    pytest.param(BigQuery, marks=pytest.mark.bigquery),
    pytest.param(Impala, marks=pytest.mark.impala)
]

if sys.version_info.major > 2:
    params_backend.append(pytest.param(MapD, marks=pytest.mark.mapd))


@pytest.fixture(params=params_backend, scope='session')
def backend(request, data_directory):
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    return backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes()


@pytest.fixture(scope='session')
def sorted_alltypes(alltypes):
    return alltypes.sort_by('id')


@pytest.fixture(scope='session')
def batting(backend):
    return backend.batting()


@pytest.fixture(scope='session')
def awards_players(backend):
    return backend.awards_players()


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes.groupby('string_col').order_by('id')


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def sorted_df(df):
    return df.sort_values('id').reset_index(drop=True)


@pytest.fixture(scope='session')
def batting_df(batting):
    return batting.execute(limit=None)


@pytest.fixture(scope='session')
def awards_players_df(awards_players):
    return awards_players.execute(limit=None)
