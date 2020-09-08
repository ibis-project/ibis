import inspect
import os
import warnings

import pytest

import ibis
import ibis.util as util
from ibis import options
from ibis.tests.expr.mocks import MockConnection


def isproperty(obj):
    return isinstance(obj, property)


class IbisTestEnv:
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
            'IBIS_TEST_TMP_DB', 'ibis_testing_tmp_db'
        )
        return tmp_db

    options.impala.temp_hdfs_path = tmp_dir = os.environ.get(
        'IBIS_TEST_TMP_HDFS_DIR', '/tmp/__ibis_test_{}'.format(util.guid())
    )

    @property
    def test_data_db(self):
        return os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')

    @property
    def test_data_dir(self):
        return os.environ.get(
            'IBIS_TEST_DATA_HDFS_DIR', '/__ibis/ibis-testing-data'
        )

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

    client = ibis.hdfs_connect(
        host=env.nn_host,
        port=env.webhdfs_port,
        auth_mechanism=env.auth_mechanism,
        verify=env.auth_mechanism not in {'GSSAPI', 'LDAP'},
        user=env.webhdfs_user,
    )

    if not client.exists(tmp_dir):
        client.mkdir(tmp_dir)
    client.chmod(tmp_dir, '777')
    return client


@pytest.fixture(scope='session')
def con_no_hdfs(env, test_data_db):
    con = ibis.impala.connect(
        host=env.impala_host,
        database=test_data_db,
        port=env.impala_port,
        auth_mechanism=env.auth_mechanism,
    )
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    try:
        yield con
    finally:
        con.set_database(test_data_db)


@pytest.fixture(scope='session')
def con(env, hdfs, test_data_db):
    con = ibis.impala.connect(
        host=env.impala_host,
        database=test_data_db,
        port=env.impala_port,
        auth_mechanism=env.auth_mechanism,
        hdfs_client=hdfs,
    )
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    try:
        yield con
    finally:
        con.set_database(test_data_db)


@pytest.fixture(scope='session')
def temp_char_table(con):
    statement = """\
CREATE TABLE IF NOT EXISTS {} (
  `group1` varchar(10),
  `group2` char(10)
)"""
    name = 'testing_varchar_support'
    sql = statement.format(name)
    con.con.execute(sql)
    try:
        yield con.table(name)
    finally:
        assert con.exists_table(name), name
        con.drop_table(name)


@pytest.fixture(scope='session')
def tmp_db(env, con, test_data_db):
    impala = pytest.importorskip("impala")
    tmp_db = env.tmp_db

    if not con.exists_database(tmp_db):
        con.create_database(tmp_db)
    try:
        yield tmp_db
    finally:
        con.set_database(test_data_db)
        try:
            con.drop_database(tmp_db, force=True)
        except impala.error.HiveServer2Error:
            # The database can be dropped by another process during tear down
            # in the middle of dropping this one if tests are running in
            # parallel.
            #
            # We only care that it gets dropped before all tests are finished
            # running.
            pass


@pytest.fixture(scope='session')
def con_no_db(env, hdfs):
    con = ibis.impala.connect(
        host=env.impala_host,
        database=None,
        port=env.impala_port,
        auth_mechanism=env.auth_mechanism,
        hdfs_client=hdfs,
    )
    if not env.use_codegen:
        con.disable_codegen()
    assert con.get_options()['DISABLE_CODEGEN'] == '1'
    try:
        yield con
    finally:
        con.set_database(None)


@pytest.fixture(scope='session')
def alltypes(con, test_data_db):
    return con.database(test_data_db).functional_alltypes


@pytest.fixture(scope='session')
def alltypes_df(alltypes):
    return alltypes.execute()


def _random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


@pytest.fixture
def temp_database(con, test_data_db):
    name = _random_identifier('database')
    con.create_database(name)
    try:
        yield name
    finally:
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
def temp_parquet_table_schema():
    return ibis.schema(
        [('id', 'int32'), ('name', 'string'), ('files', 'int32')]
    )


@pytest.fixture
def temp_parquet_table(con, tmp_db, temp_parquet_table_schema):
    name = util.guid()
    db = con.database(tmp_db)
    db.create_table(name, schema=temp_parquet_table_schema, format='parquet')
    try:
        yield db[name]
    finally:
        db.client.drop_table(name, database=tmp_db)


@pytest.fixture
def temp_parquet_table2(con, tmp_db, temp_parquet_table_schema):
    name = util.guid()
    db = con.database(tmp_db)
    db.create_table(name, schema=temp_parquet_table_schema, format='parquet')
    try:
        yield db[name]
    finally:
        db.client.drop_table(name, database=tmp_db)


@pytest.fixture
def mockcon():
    return MockConnection()


@pytest.fixture(scope='session')
def kudu_table(con, test_data_db):
    name = 'kudu_backed_table'
    con.raw_sql(
        """\
CREATE TABLE {database}.{name} (
  a STRING,
  PRIMARY KEY(a)
)
PARTITION BY HASH PARTITIONS 2
STORED AS KUDU
TBLPROPERTIES (
  'kudu.master_addresses' = 'kudu',
  'kudu.num_tablet_replicas' = '1'
)""".format(
            database=test_data_db, name=name
        )
    )
    drop_sql = 'DROP TABLE {database}.{name}'.format(
        database=test_data_db, name=name
    )
    try:
        yield con.table(name)
    finally:
        con.raw_sql(drop_sql)
