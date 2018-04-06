import ibis
import os
import pytest


MAPD_HOST = os.environ.get('IBIS_TEST_MAPD_HOST', 'localhost')
MAPD_PORT = int(os.environ.get('IBIS_TEST_MAPD_PORT', 9091))
MAPD_USER = os.environ.get('IBIS_TEST_MAPD_USER', 'mapd')
MAPD_PASS = os.environ.get('IBIS_TEST_MAPD_PASSWORD', 'HyperInteractive')
MAPD_DB = os.environ.get('IBIS_TEST_DATA_DB', 'mapd')


@pytest.fixture(scope='module')
def con():
    """

    :return:
    """
    return ibis.mapd.connect(
        host=MAPD_HOST,
        port=MAPD_PORT,
        user=MAPD_USER,
        password=MAPD_PASS,
        dbname=MAPD_DB,
    )


@pytest.fixture
def translate():
    """

    :return:
    """
    from ibis.mapd.compiler import MapDDialect
    dialect = MapDDialect()
    context = dialect.make_context()
    return lambda expr: dialect.translator(expr, context).get_result()
