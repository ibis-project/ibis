from __future__ import annotations

import os
from pathlib import Path

import pytest

import ibis
import ibis.expr.types as ir
import ibis.util as util
from ibis import options
from ibis.backends.impala.compiler import ImpalaCompiler, ImpalaExprTranslator
from ibis.backends.tests.base import (
    BackendTest,
    RoundAwayFromZero,
    UnorderedComparator,
)
from ibis.tests.expr.mocks import MockBackend


class TestConf(UnorderedComparator, BackendTest, RoundAwayFromZero):
    supports_arrays = True
    supports_arrays_outside_of_select = False
    check_dtype = False
    supports_divide_by_zero = True
    returned_timestamp_unit = 's'

    @staticmethod
    def connect(data_directory: Path):
        fsspec = pytest.importorskip("fsspec")

        env = IbisTestEnv()
        return ibis.impala.connect(
            host=env.impala_host,
            port=env.impala_port,
            auth_mechanism=env.auth_mechanism,
            hdfs_client=fsspec.filesystem(
                env.hdfs_protocol,
                host=env.nn_host,
                port=env.hdfs_port,
                user=env.hdfs_user,
            ),
            database=env.test_data_db,
        )

    def _get_original_column_names(self, tablename: str) -> list[str]:
        import pyarrow.parquet as pq

        pq_file = pq.ParquetFile(
            self.data_directory
            / "parquet"
            / tablename
            / f"{tablename}.parquet"
        )
        return pq_file.schema.names

    def _get_renamed_table(self, tablename: str) -> ir.TableExpr:
        t = self.connection.table(tablename)
        original_names = self._get_original_column_names(tablename)
        return t.relabel(dict(zip(t.columns, original_names)))

    @property
    def batting(self) -> ir.TableExpr:
        return self._get_renamed_table("batting")

    @property
    def awards_players(self) -> ir.TableExpr:
        return self._get_renamed_table("awards_players")


class IbisTestEnv:
    def __init__(self):
        if options.impala is None:
            ibis.backends.impala.Backend.register_options()

    @property
    def impala_host(self):
        return os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')

    @property
    def impala_port(self):
        return int(os.environ.get('IBIS_TEST_IMPALA_PORT', "21050"))

    @property
    def tmp_db(self):
        options.impala.temp_db = tmp_db = os.environ.get(
            'IBIS_TEST_TMP_DB', 'ibis_testing_tmp_db'
        )
        return tmp_db

    @property
    def tmp_dir(self):
        options.impala.temp_hdfs_path = tmp_dir = os.environ.get(
            'IBIS_TEST_TMP_HDFS_DIR', f'/tmp/__ibis_test_{util.guid()}'
        )
        return tmp_dir

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
    def hdfs_port(self):
        return int(os.environ.get('IBIS_TEST_HDFS_PORT', 50070))

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
    def hdfs_user(self):
        return os.environ.get('IBIS_TEST_HDFS_USER', 'hdfs')

    @property
    def hdfs_protocol(self):
        return os.environ.get("IBIS_TEST_HDFS_PROTOCOL", "webhdfs")


@pytest.fixture
def env():
    return IbisTestEnv()


@pytest.fixture
def tmp_dir(env):
    options.impala.temp_hdfs_path = tmp_dir = env.tmp_dir
    return tmp_dir


@pytest.fixture
def test_data_db(env):
    return env.test_data_db


@pytest.fixture
def test_data_dir(env):
    return env.test_data_dir


@pytest.fixture
def hdfs(env, tmp_dir):
    fsspec = pytest.importorskip("fsspec")
    client = fsspec.filesystem(
        env.hdfs_protocol,
        host=env.nn_host,
        port=env.hdfs_port,
        user=env.hdfs_user,
    )

    if not client.exists(tmp_dir):
        client.mkdir(tmp_dir)
    return client


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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
        assert name in con.list_tables(), name
        con.drop_table(name)


@pytest.fixture
def tmp_db(env, con, test_data_db):
    impala = pytest.importorskip("impala")

    tmp_db = env.tmp_db

    if tmp_db not in con.list_databases():
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


@pytest.fixture
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


@pytest.fixture
def alltypes(con, test_data_db):
    return con.table("functional_alltypes")


@pytest.fixture
def alltypes_df(alltypes):
    return alltypes.execute()


def _random_identifier(suffix):
    return f'__ibis_test_{suffix}_{util.guid()}'


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
        assert name in con.list_tables(), name
        con.drop_table(name)


@pytest.fixture
def temp_table_db(con, temp_database):
    name = _random_identifier('table')
    try:
        yield temp_database, name
    finally:
        assert name in con.list_tables(database=temp_database), name
        con.drop_table(name, database=temp_database)


@pytest.fixture
def temp_view(con):
    name = _random_identifier('view')
    try:
        yield name
    finally:
        assert name in con.list_tables(), name
        con.drop_view(name)


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


@pytest.fixture(scope="session")
def mockcon():
    return MockBackend()


@pytest.fixture
def kudu_table(con, test_data_db):
    name = 'kudu_backed_table'
    con.raw_sql(
        f"""
CREATE TABLE {test_data_db}.{name} (
  a STRING,
  PRIMARY KEY(a)
)
PARTITION BY HASH PARTITIONS 2
STORED AS KUDU
TBLPROPERTIES (
  'kudu.master_addresses' = 'kudu',
  'kudu.num_tablet_replicas' = '1'
)"""
    )
    drop_sql = f'DROP TABLE {test_data_db}.{name}'
    try:
        yield con.table(name)
    finally:
        con.raw_sql(drop_sql)


def translate(expr, context=None, named=False):
    if context is None:
        context = ImpalaCompiler.make_context()
    translator = ImpalaExprTranslator(expr, context=context, named=named)
    return translator.get_result()
