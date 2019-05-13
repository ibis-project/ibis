import os

import pytest

import ibis
import ibis.util as util

OMNISCI_HOST = os.environ.get('IBIS_TEST_OMNISCI_HOST', 'localhost')
OMNISCI_PORT = int(os.environ.get('IBIS_TEST_OMNISCI_PORT', 6274))
OMNISCI_USER = os.environ.get('IBIS_TEST_OMNISCI_USER', 'mapd')
OMNISCI_PASS = os.environ.get('IBIS_TEST_OMNISCI_PASSWORD', 'HyperInteractive')
OMNISCI_DB = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')


@pytest.fixture(scope='module')
def con():
    return ibis.mapd.connect(
        host=OMNISCI_HOST,
        port=OMNISCI_PORT,
        user=OMNISCI_USER,
        password=OMNISCI_PASS,
        database=OMNISCI_DB,
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


@pytest.fixture(scope='module')
def df_alltypes(alltypes):
    return alltypes.execute()


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
    return OMNISCI_DB


@pytest.fixture
def temp_database(con, test_data_db):
    name = _random_identifier('database')
    con.create_database(name)
    try:
        yield name
    finally:
        con.set_database(test_data_db)
        con.drop_database(name, force=True)
