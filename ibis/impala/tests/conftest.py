import os

import pytest

from ibis import options
import ibis.util as util
import ibis

from ibis.expr.tests.mocks import MockConnection


class IbisTestEnv(object):
    @property
    def impala_host(self):
        return os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')

    @property
    def impala_port(self):
        return int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))

    @property
    def tmp_db(self):
        options.impala.temp_db = tmp_db = os.environ.get(
                'IBIS_TEST_TMP_DB', '__ibis_tmp_{}'.format(util.guid()))
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
    def cleanup_test_data(self):
        return os.environ.get(
            'IBIS_TEST_CLEANUP_TEST_DATA', 'True').lower() == 'true'

    @property
    def auth_mechanism(self):
        return os.environ.get('IBIS_TEST_AUTH_MECH', 'NOSASL')

    @property
    def llvm_config(self):
        return os.environ.get('IBIS_TEST_LLVM_CONFIG', None)

    @property
    def webhdfs_user(self):
        return os.environ.get('IBIS_TEST_WEBHDFS_USER', 'hdfs')


@pytest.fixture
def impala_host():
    return os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')


@pytest.fixture
def impala_port():
    return int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))


@pytest.fixture
def tmp_db():
    options.impala.temp_db = tmp_db = os.environ.get(
        'IBIS_TEST_TMP_DB', '__ibis_tmp_{}'.format(util.guid()))
    return tmp_db


@pytest.fixture(scope='session')
def tmp_dir():
    options.impala.temp_hdfs_path = tmp_dir = os.environ.get(
        'IBIS_TEST_TMP_HDFS_DIR', '/tmp/__ibis_test_{}'.format(util.guid()))
    return tmp_dir


@pytest.fixture
def test_data_db():
    return os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')


@pytest.fixture
def test_data_dir():
    return os.environ.get(
        'IBIS_TEST_DATA_HDFS_DIR', '/__ibis/ibis-testing-data')


@pytest.fixture
def nn_host():
    return os.environ.get('IBIS_TEST_NN_HOST', 'localhost')


@pytest.fixture
def webhdfs_port():
    # 5070 is default for impala dev env
    return int(os.environ.get('IBIS_TEST_WEBHDFS_PORT', 50070))


@pytest.fixture
def hdfs_superuser():
    return os.environ.get('IBIS_TEST_HDFS_SUPERUSER', 'hdfs')


@pytest.fixture
def use_codegen():
    return os.environ.get('IBIS_TEST_USE_CODEGEN', 'False').lower() == 'true'


@pytest.fixture
def cleanup_test_data():
    return os.environ.get(
        'IBIS_TEST_CLEANUP_TEST_DATA', 'True').lower() == 'true'


@pytest.fixture
def auth_mechanism():
    return os.environ.get('IBIS_TEST_AUTH_MECH', 'NOSASL')


@pytest.fixture
def llvm_config():
    return os.environ.get('IBIS_TEST_LLVM_CONFIG', None)


@pytest.fixture
def webhdfs_user():
    return os.environ.get('IBIS_TEST_WEBHDFS_USER', 'hdfs')


@pytest.fixture
def hdfs(nn_host, webhdfs_port, auth_mechanism, webhdfs_user, tmp_dir):
    if auth_mechanism in ['GSSAPI', 'LDAP']:
        print("Warning: ignoring invalid Certificate Authority errors")

    client = ibis.hdfs_connect(host=nn_host,
                               port=webhdfs_port,
                               auth_mechanism=auth_mechanism,
                               verify=auth_mechanism not in ['GSSAPI', 'LDAP'],
                               user=webhdfs_user)

    if not client.exists(tmp_dir):
        client.mkdir(tmp_dir)
    client.chmod(tmp_dir, '777')
    return client


@pytest.fixture
def con_no_hdfs(
    impala_host, test_data_db, impala_port, auth_mechanism, use_codegen
):
    con = ibis.impala.connect(host=impala_host,
                              database=test_data_db,
                              port=impala_port,
                              auth_mechanism=auth_mechanism)
    if not use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    return con


@pytest.fixture
def con(
    hdfs, impala_host, test_data_db, impala_port, auth_mechanism, use_codegen,
    tmp_db
):
    con = ibis.impala.connect(host=impala_host,
                              database=test_data_db,
                              port=impala_port,
                              auth_mechanism=auth_mechanism,
                              hdfs_client=hdfs)
    if not use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'

    if not con.exists_database(tmp_db):
        con.create_database(tmp_db)
    try:
        yield con
    finally:
        con.drop_database(tmp_db, force=True)


@pytest.fixture
def con_no_db(hdfs, impala_host, impala_port, auth_mechanism, use_codegen):
    con = ibis.impala.connect(host=impala_host,
                              database=None,
                              port=impala_port,
                              auth_mechanism=auth_mechanism,
                              hdfs_client=hdfs)
    if not use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    return con


@pytest.fixture
def alltypes(con):
    return con.table('functional_alltypes')


@pytest.fixture
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture
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
        yield '{}.{}'.format(temp_database, name)
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
        yield '{}.{}'.format(temp_database, name)
    finally:
        assert con.exists_table(name, database=temp_database), name
        con.drop_view(name, database=temp_database)


@pytest.fixture
def mockcon():
    return MockConnection()
