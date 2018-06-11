import inspect
import os
import warnings

import pytest

import ibis.util as util
import ibis

from ibis import options
from ibis.compat import map, zip
from ibis.expr.tests.mocks import MockConnection


def isproperty(obj):
    return isinstance(obj, property)


class IbisTestEnv(object):
    def items(self):
        return [
            (name, getattr(self, name))
            for name, _ in inspect.getmembers(type(self), predicate=isproperty)
        ]

    def __repr__(self):
        lines = map('{}={!r},'.format, *zip(*self.items()))
        return '{}(\n{}\n)'.format(
            type(self).__name__, util.indent('\n'.join(lines), 4)
        )

    @property
    def impala_host(self):
        return os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')

    @property
    def impala_port(self):
        return int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))

    @property
    def tmp_db(self):
        options.impala.temp_db = tmp_db = os.environ.get(
                'IBIS_TEST_TMP_DB', 'ibis_testing_tmp_db')
        return tmp_db

    options.impala.temp_hdfs_path = tmp_dir = os.environ.get(
        'IBIS_TEST_TMP_HDFS_DIR', '/tmp/__ibis_test_{}'.format(util.guid()))

    @property
    def test_data_db(self):
        return os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')

    @property
    def test_data_dir(self):
        return os.environ.get(
            'IBIS_TEST_DATA_HDFS_DIR', '/__ibis/ibis-testing-data')

    @property
    def nn_host(self):
        return os.environ.get('IBIS_TEST_NN_HOST', 'localhost')

    @property
    def webhdfs_port(self):
        # 5070 is default for impala dev env
        return int(os.environ.get('IBIS_TEST_WEBHDFS_PORT', 50070))

    @property
    def hdfs_superuser(self):
        return os.environ.get('IBIS_TEST_HDFS_SUPERUSER', 'hdfs')

    @property
    def use_codegen(self):
        return (
            os.environ.get('IBIS_TEST_USE_CODEGEN', 'False').lower() == 'true'
        )

    @property
    def auth_mechanism(self):
        return os.environ.get('IBIS_TEST_AUTH_MECH', 'NOSASL')

    @property
    def webhdfs_user(self):
        return os.environ.get('IBIS_TEST_WEBHDFS_USER', 'hdfs')


@pytest.fixture(scope='session')
def env():
    return IbisTestEnv()


@pytest.fixture(scope='session')
def tmp_dir(env):
    options.impala.temp_hdfs_path = tmp_dir = env.tmp_dir
    return tmp_dir


@pytest.fixture(scope='session')
def test_data_db(env):
    return env.test_data_db


@pytest.fixture(scope='session')
def test_data_dir(env):
    return env.test_data_dir


@pytest.fixture(scope='session')
def hdfs_superuser(env):
    return env.hdfs_superuser
    return os.environ.get('IBIS_TEST_HDFS_SUPERUSER', 'hdfs')


@pytest.fixture(scope='session')
def hdfs(env, tmp_dir):
    pytest.importorskip('requests')

    if env.auth_mechanism in {'GSSAPI', 'LDAP'}:
        warnings.warn("Ignoring invalid Certificate Authority errors")

    client = ibis.hdfs_connect(host=env.nn_host,
                               port=env.webhdfs_port,
                               auth_mechanism=env.auth_mechanism,
                               verify=env.auth_mechanism not in {
                                   'GSSAPI', 'LDAP'
                               },
                               user=env.webhdfs_user)

    if not client.exists(tmp_dir):
        client.mkdir(tmp_dir)
    client.chmod(tmp_dir, '777')
    return client


@pytest.fixture(scope='session')
def con_no_hdfs(env, test_data_db):
    con = ibis.impala.connect(host=env.impala_host,
                              database=test_data_db,
                              port=env.impala_port,
                              auth_mechanism=env.auth_mechanism)
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    return con


@pytest.fixture(scope='session')
def con(env, hdfs, test_data_db):
    con = ibis.impala.connect(host=env.impala_host,
                              database=test_data_db,
                              port=env.impala_port,
                              auth_mechanism=env.auth_mechanism,
                              hdfs_client=hdfs)
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    return con


@pytest.fixture
def temp_char_table(con, tmp_db):
    statement = """\
CREATE TABLE IF NOT EXISTS {} (
  `group1` varchar(10),
  `group2` char(10)
)"""
    name = 'testing_varchar_support'
    sql = statement.format(name)
    con.con.execute(sql)
    while not con.exists_table(name, database=tmp_db):
        pass
    try:
        yield con.database(tmp_db)[name]
    finally:
        con.drop_table(name, database=tmp_db)


@pytest.fixture(scope='session')
def tmp_db(env, con):
    tmp_db = env.tmp_db
    if not con.exists_database(tmp_db):
        con.create_database(tmp_db)
    try:
        yield tmp_db
    finally:
        assert con.exists_database(tmp_db), tmp_db
        con.drop_database(tmp_db, force=True)


@pytest.fixture(scope='session')
def con_no_db(env, hdfs):
    con = ibis.impala.connect(host=env.impala_host,
                              database=None,
                              port=env.impala_port,
                              auth_mechanism=env.auth_mechanism,
                              hdfs_client=hdfs)
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    return con


@pytest.fixture(scope='session')
def alltypes(con):
    return con.table('functional_alltypes')


@pytest.fixture(scope='session')
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def db(con, test_data_db):
    return con.database(test_data_db)


def _random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


@pytest.fixture
def temp_database(con, test_data_db):
    name = _random_identifier('database')
    con.create_database(name)
    try:
        yield name
    finally:
        assert con.exists_database(name), name
        con.set_database(test_data_db)
        con.drop_database(name, force=True)


@pytest.fixture
def temp_table(con):
    name = _random_identifier('table')
    try:
        yield name
    finally:
        assert con.exists_table(name), name
        con.drop_table(name)


@pytest.fixture
def temp_table_db(con, temp_database):
    name = _random_identifier('table')
    try:
        yield temp_database, name
    finally:
        assert con.exists_table(name, database=temp_database), name
        con.drop_table(name, database=temp_database)


@pytest.fixture
def temp_view(con):
    name = _random_identifier('view')
    try:
        yield name
    finally:
        assert con.exists_table(name), name
        con.drop_view(name)


@pytest.fixture
def temp_view_db(con, temp_database):
    name = _random_identifier('view')
    try:
        yield temp_database, name
    finally:
        assert con.exists_table(name, database=temp_database), name
        con.drop_view(name, database=temp_database)


@pytest.fixture
def mockcon():
    return MockConnection()
