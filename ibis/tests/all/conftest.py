import pytest

from ibis.tests.backends import (Csv, Parquet, SQLite, Postgres, Clickhouse,
                                 Impala)


pytest.mark.backend()


@pytest.fixture(params=[
    Csv,
    Parquet,
    SQLite,
    Postgres,
    Clickhouse,
    Impala
], scope='session')
def backend(request, data_directory):
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    return backend.connection
