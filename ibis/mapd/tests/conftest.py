import ibis
import os
import pytest


MAPD_HOST = os.environ.get('IBIS_TEST_MAPD_HOST', 'localhost')
MAPD_PORT = int(os.environ.get('IBIS_TEST_MAPD_PORT', 9091))
MAPD_USER = os.environ.get('IBIS_TEST_MAPD_USER', 'mapd')
MAPD_PASS = os.environ.get('IBIS_TEST_MAPD_PASSWORD', 'HyperInteractive')
MAPD_DB = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')


@pytest.fixture(scope='module')
def con():
    if MAPD_DB != 'mapd':
        import pymapd
        conn = pymapd.connect(
            host=MAPD_HOST,
            user=MAPD_USER,
            password=MAPD_PASS,
            port=MAPD_PORT, dbname='mapd'
        )
        try:
            conn.execute('CREATE DATABASE {}'.format(MAPD_DB))
        except Exception as e:
            print(e)
        conn.close()

    return ibis.mapd.connect(
        host=MAPD_HOST,
        port=MAPD_PORT,
        user=MAPD_USER,
        password=MAPD_PASS,
        database=MAPD_DB,
    )


@pytest.fixture(scope='module')
def alltypes(con):
    return con.table('functional_alltypes')


@pytest.fixture
def translate():
    """

    :return:
    """
    from ibis.mapd.compiler import MapDDialect
    dialect = MapDDialect()
    context = dialect.make_context()
    return lambda expr: dialect.translator(expr, context).get_result()
