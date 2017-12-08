import os
import pytest

from functools import partial

import ibis


defaults = dict(
    clickhouse=dict(
        host='localhost',
        port=9000,
        user='default',
        password='',
        database='ibis_testing'
    ),
    impala=dict(
        host='localhost',
    ),
    bigquery=dict(
        project_id=None,
        dataset_id='testing'
    ),
    postgres=dict(
        host='localhost',
        port=5432,
        user='postgres',
        password='ibis',
        database='ibis_testing'
    ),
    sqlite=dict(
        database='testing/ibis_testing.db'
    )
)


def env(backend):
    key = partial('IBIS_{}_{}'.format, backend.upper())
    return {k: os.environ.get(key(k), v) for k, v in defaults[backend].items()}


@pytest.fixture(params=[
    'clickhouse',
    # 'impala',
    # 'bigquery',
    'postgres',
    # 'sqlite'
])
def client(request):
    name = request.param
    config = env(name)
    try:
        backend = getattr(ibis, name)
        return backend.connect(**config)
    except ImportError:
        return pytest.skip('Backend {} needs to be installed'.format(name))


@pytest.fixture
def alltypes(client):
    return client.table('functional_alltypes')
