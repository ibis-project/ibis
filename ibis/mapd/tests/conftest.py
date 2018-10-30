import ibis
import ibis.util as util
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


@pytest.fixture(scope='module')
def awards_players(con):
    return con.table('awards_players')


@pytest.fixture(scope='module')
def batting(con):
    return con.table('batting')


@pytest.fixture
def translate():
    """

    :return:
    """
    from ibis.mapd.compiler import MapDDialect
    dialect = MapDDialect()
    context = dialect.make_context()
    return lambda expr: dialect.translator(expr, context).get_result()


def _random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


@pytest.fixture
def temp_table(con):
    name = _random_identifier('table')
    try:
        yield name
    finally:
        assert con.exists_table(name), name
        con.drop_table(name)


@pytest.fixture(scope='session')
def test_data_db():
    return MAPD_DB


@pytest.fixture
def temp_database(con, test_data_db):
    name = _random_identifier('database')
    con.create_database(name)
    try:
        yield name
    finally:
        con.set_database(test_data_db)
        con.drop_database(name, force=True)
