#! /usr/bin/env python
# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import getpass
import os
import shutil
import tempfile
import os.path as osp
from os.path import join as pjoin
from subprocess import check_call

from click import group, option
import sqlalchemy as sa
from sqlalchemy import create_engine

import ibis
from ibis.compat import BytesIO
from ibis.common import IbisError
from ibis.impala.tests.common import IbisTestEnv
from ibis.util import guid

import numpy as np

import pandas as pd
import pandas.util.testing as tm

ENV = IbisTestEnv()
IBIS_TEST_DATA_S3_BUCKET = 'ibis-resources'
IBIS_TEST_DATA_LOCAL_DIR = 'ibis-testing-data'

TARBALL_NAME = 'ibis-testing-data.tar.gz'
IBIS_TEST_DATA_TARBALL = 'testing/{0}'.format(TARBALL_NAME)


IBIS_TEST_AWS_KEY_ID = os.environ.get('IBIS_TEST_AWS_KEY_ID')
IBIS_TEST_AWS_SECRET = os.environ.get('IBIS_TEST_AWS_SECRET')


def make_ibis_client():
    hc = ibis.hdfs_connect(host=ENV.nn_host, port=ENV.webhdfs_port,
                           auth_mechanism=ENV.auth_mechanism,
                           verify=(ENV.auth_mechanism
                                   not in ['GSSAPI', 'LDAP']))
    if ENV.auth_mechanism in ['GSSAPI', 'LDAP']:
        print("Warning: ignoring invalid Certificate Authority errors")
    return ibis.impala.connect(host=ENV.impala_host, port=ENV.impala_port,
                               auth_mechanism=ENV.auth_mechanism,
                               hdfs_client=hc)


def can_write_to_hdfs(con):
    test_path = pjoin(ENV.test_data_dir, ibis.util.guid())
    test_file = BytesIO(ibis.util.guid().encode('utf-8'))
    try:
        con.hdfs.put(test_path, test_file)
        con.hdfs.rm(test_path)
        return True
    except:
        return False


def can_build_udfs():
    try:
        check_call('which cmake', shell=True)
    except:
        print('Could not find cmake on PATH')
        return False
    try:
        check_call('which make', shell=True)
    except:
        print('Could not find make on PATH')
        return False
    try:
        check_call('which clang++', shell=True)
    except:
        print('Could not find LLVM on PATH; if IBIS_TEST_LLVM_CONFIG is set, '
              'try setting PATH="$($IBIS_TEST_LLVM_CONFIG --bindir):$PATH"')
        return False
    return True


def is_impala_loaded(con):
    if not con.hdfs.exists(ENV.test_data_dir):
        return False
    if not con.exists_database(ENV.test_data_db):
        return False
    return True


def is_udf_loaded(con):
    bitcode_dir = pjoin(ENV.test_data_dir, 'udf')
    if con.hdfs.exists(bitcode_dir):
        return True
    return False


def dnload_ibis_test_data_from_s3(local_path):
    url = 'https://{0}.s3.amazonaws.com/{1}'.format(
        IBIS_TEST_DATA_S3_BUCKET, IBIS_TEST_DATA_TARBALL)
    cmd = 'cd {0} && wget -q {1} && tar -xzf {2}'.format(
        local_path, url, TARBALL_NAME)
    check_call(cmd, shell=True)
    data_dir = pjoin(local_path, IBIS_TEST_DATA_LOCAL_DIR)
    print('Downloaded {0} and unpacked it to {1}'.format(url, data_dir))
    return data_dir


def upload_ibis_test_data_to_hdfs(con, data_path):
    hdfs = con.hdfs
    if hdfs.exists(ENV.test_data_dir):
        hdfs.rmdir(ENV.test_data_dir)
    hdfs.put(ENV.test_data_dir, data_path, verbose=True)


def create_test_database(con):
    if con.exists_database(ENV.test_data_db):
        con.drop_database(ENV.test_data_db, force=True)
    con.create_database(ENV.test_data_db)
    print('Created database {0}'.format(ENV.test_data_db))


def create_parquet_tables(con):
    parquet_files = con.hdfs.ls(pjoin(ENV.test_data_dir, 'parquet'))
    schemas = {
        'functional_alltypes': ibis.schema(
            [('id', 'int32'),
             ('bool_col', 'boolean'),
             ('tinyint_col', 'int8'),
             ('smallint_col', 'int16'),
             ('int_col', 'int32'),
             ('bigint_col', 'int64'),
             ('float_col', 'float'),
             ('double_col', 'double'),
             ('date_string_col', 'string'),
             ('string_col', 'string'),
             ('timestamp_col', 'timestamp'),
             ('year', 'int32'),
             ('month', 'int32')]),
        'tpch_region': ibis.schema(
            [('r_regionkey', 'int16'),
             ('r_name', 'string'),
             ('r_comment', 'string')])}
    tables = []
    for table_name in parquet_files:
        print('Creating {0}'.format(table_name))
        # if no schema infer!
        schema = schemas.get(table_name)
        path = pjoin(ENV.test_data_dir, 'parquet', table_name)
        table = con.parquet_file(path, schema=schema, name=table_name,
                                 database=ENV.test_data_db, persist=True)
        tables.append(table)
    return tables


def create_avro_tables(con):
    avro_files = con.hdfs.ls(pjoin(ENV.test_data_dir, 'avro'))
    schemas = {
        'tpch_region_avro': {
            'type': 'record',
            'name': 'a',
            'fields': [
                {'name': 'R_REGIONKEY', 'type': ['null', 'int']},
                {'name': 'R_NAME', 'type': ['null', 'string']},
                {'name': 'R_COMMENT', 'type': ['null', 'string']}]}}
    tables = []
    for table_name in avro_files:
        print('Creating {0}'.format(table_name))
        schema = schemas[table_name]
        path = pjoin(ENV.test_data_dir, 'avro', table_name)
        table = con.avro_file(path, schema, name=table_name,
                              database=ENV.test_data_db, persist=True)
        tables.append(table)
    return tables


def build_udfs():
    print('Building UDFs')
    ibis_home_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    udf_dir = pjoin(ibis_home_dir, 'testing', 'udf')
    check_call('cmake . && make VERBOSE=1', shell=True, cwd=udf_dir)


def upload_udfs(con):
    ibis_home_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    build_dir = pjoin(ibis_home_dir, 'testing', 'udf', 'build')
    bitcode_dir = pjoin(ENV.test_data_dir, 'udf')
    print('Uploading UDFs to {0}'.format(bitcode_dir))
    if con.hdfs.exists(bitcode_dir):
        con.hdfs.rmdir(bitcode_dir)
    con.hdfs.put(bitcode_dir, build_dir, verbose=True)


def scrape_parquet_files(tmp_db, con):
    to_scrape = [('tpch', x) for x in con.list_tables(database='tpch')]
    to_scrape.append(('functional', 'alltypes'))
    for db, tname in to_scrape:
        table = con.table(tname, database=db)
        new_name = '{0}_{1}'.format(db, tname)
        print('Creating {0}'.format(new_name))
        con.create_table(new_name, table, database=tmp_db)


def download_parquet_files(con, tmp_db_hdfs_path):
    parquet_path = pjoin(IBIS_TEST_DATA_LOCAL_DIR, 'parquet')
    print("Downloading {0}".format(parquet_path))
    con.hdfs.get(tmp_db_hdfs_path, parquet_path)


def get_postgres_engine():
    pg_user = os.environ.get('IBIS_POSTGRES_USER', getpass.getuser())
    pg_pass = os.environ.get('IBIS_POSTGRES_PASS')

    if pg_pass:
        creds = '{0}:{1}'.format(pg_user, pg_pass)
    else:
        creds = pg_user

    engine = (create_engine('postgresql://{0}@localhost/ibis_testing'
                            .format(creds)))
    return engine


def get_sqlite_engine():
    path = pjoin(IBIS_TEST_DATA_LOCAL_DIR, 'ibis_testing.db')
    return create_engine('sqlite:///{0}'.format(path))


def load_sql_databases(con, engines):
    csv_path = guid()

    generate_sql_csv_sources(csv_path, con.database('ibis_testing'))
    for engine in engines:
        make_testing_db(csv_path, engine)
    shutil.rmtree(csv_path)


def download_avro_files(con):
    avro_hdfs_path = '/test-warehouse/tpch.region_avro'
    avro_local_path = pjoin(IBIS_TEST_DATA_LOCAL_DIR, 'avro')
    os.mkdir(avro_local_path)
    print("Downloading {0}".format(avro_hdfs_path))
    con.hdfs.get(avro_hdfs_path, pjoin(avro_local_path, 'tpch_region_avro'))


def generate_csv_files():
    N = 10
    nfiles = 10
    df = pd.DataFrame({'foo': [tm.rands(10) for _ in xrange(N)],
                       'bar': np.random.randn(N),
                       'baz': np.random.randint(0, 100, size=N)},
                      columns=['foo', 'bar', 'baz'])
    csv_base = pjoin(IBIS_TEST_DATA_LOCAL_DIR, 'csv')
    os.mkdir(csv_base)
    for i in xrange(nfiles):
        csv_path = pjoin(csv_base, '{0}.csv'.format(i))
        print('Writing {0}'.format(csv_path))
        df.to_csv(csv_path, index=False, header=False)


def copy_tarball_to_versioned_backup(bucket):
    key = bucket.get_key(IBIS_TEST_DATA_TARBALL)
    if key:
        names = [k.name for k in bucket.list(prefix=IBIS_TEST_DATA_TARBALL)]
        names.remove(IBIS_TEST_DATA_TARBALL)
        # get the highest number for this key name
        last = sorted([int(names.split('.')[-1]) for name in names])[-1]
        next_key = '{0}.{1}'.format(IBIS_TEST_DATA_TARBALL, last + 1)
        key.copy(IBIS_TEST_DATA_S3_BUCKET, next_key)
        key.delete()
    assert bucket.get_key(IBIS_TEST_DATA_TARBALL) is None


_sql_tpch_tables = ['tpch_lineitem', 'tpch_customer',
                    'tpch_region', 'tpch_nation', 'tpch_orders']

_sql_tables = ['functional_alltypes']


def _project_tpch_lineitem(t):
    return t['l_orderkey',
             'l_partkey',
             'l_suppkey',
             'l_linenumber',
             t.l_quantity.cast('double'),
             t.l_extendedprice.cast('double'),
             t.l_discount.cast('double'),
             t.l_tax.cast('double'),
             'l_returnflag',
             'l_linestatus',
             'l_shipdate',
             'l_commitdate',
             'l_receiptdate',
             'l_shipinstruct',
             'l_shipmode']


def _project_tpch_orders(t):
    return t['o_orderkey',
             'o_custkey',
             'o_orderstatus',
             t.o_totalprice.cast('double'),
             'o_orderdate',
             'o_orderpriority',
             'o_clerk',
             'o_shippriority']


def _project_tpch_customer(t):
    return t['c_custkey',
             'c_name',
             'c_nationkey',
             'c_phone',
             'c_acctbal',
             'c_mktsegment']


_projectors = {
    'tpch_customer': _project_tpch_customer,
    'tpch_lineitem': _project_tpch_lineitem,
    'tpch_orders': _project_tpch_orders,
}


def generate_sql_csv_sources(output_path, db):
    ibis.options.sql.default_limit = None

    if not osp.exists(output_path):
        os.mkdir(output_path)

    for name in _sql_tables:
        print(name)
        table = db[name]

        if name in _projectors:
            table = _projectors[name](table)

        df = table.execute()
        path = osp.join(output_path, name)
        df.to_csv('{0}.csv'.format(path), na_rep='\\N')


def make_testing_db(csv_dir, con):
    for name in _sql_tables:
        print(name)
        path = osp.join(csv_dir, '{0}.csv'.format(name))
        df = pd.read_csv(path, na_values=['\\N'], dtype={'bool_col': 'bool'})
        df.to_sql(
            name,
            con,
            chunksize=10000,
            if_exists='replace',
            dtype={
                'index': sa.INTEGER,
                'id': sa.INTEGER,
                'bool_col': sa.BOOLEAN,
                'tinyint_col': sa.SMALLINT,
                'smallint_col': sa.SMALLINT,
                'int_col': sa.INTEGER,
                'bigint_col': sa.BIGINT,
                'float_col': sa.REAL,
                'double_col': sa.FLOAT,
                'date_string_col': sa.TEXT,
                'string_col': sa.TEXT,
                'timestamp_col': sa.TIMESTAMP,
                'year': sa.INTEGER,
                'month': sa.INTEGER,
            }
        )


# ==========================================


@group(context_settings={'help_option_names': ['-h', '--help']})
def main():
    """Manage test data for Ibis"""
    pass


@main.command()
def printenv():
    """Print current IbisTestEnv"""
    print(str(ENV))


@main.command()
@option('--create-tarball', is_flag=True,
        help="Create a gzipped tarball")
@option('--push-to-s3', is_flag=True,
        help="Also push the tarball to s3://ibis-test-resources")
def create(create_tarball, push_to_s3):
    """Create Ibis test data"""
    print(str(ENV))

    con = make_ibis_client()

    # verify some assumptions before proceeding
    if push_to_s3 and not create_tarball:
        raise IbisError(
            "Must specify --create-tarball if specifying --push-to-s3")
    if osp.exists(IBIS_TEST_DATA_LOCAL_DIR):
        raise IbisError(
            'Local dir {0} already exists; please remove it first'.format(
                IBIS_TEST_DATA_LOCAL_DIR))
    if not con.exists_database('tpch'):
        raise IbisError('`tpch` database does not exist')
    if not con.hdfs.exists('/test-warehouse/tpch.region_avro'):
        raise IbisError(
            'HDFS dir /test-warehouse/tpch.region_avro does not exist')

    # generate tmp identifiers
    tmp_db_hdfs_path = pjoin(ENV.tmp_dir, guid())
    tmp_db = guid()
    os.mkdir(IBIS_TEST_DATA_LOCAL_DIR)
    try:
        # create the tmp data locally
        con.create_database(tmp_db, path=tmp_db_hdfs_path)
        print('Created database {0} at {1}'.format(tmp_db, tmp_db_hdfs_path))

        # create the local data set
        scrape_parquet_files(tmp_db, con)
        download_parquet_files(con, tmp_db_hdfs_path)
        download_avro_files(con)
        generate_csv_files()

        # Only populate SQLite here
        engines = [get_sqlite_engine()]
        load_sql_databases(con, engines)
    finally:
        con.drop_database(tmp_db, force=True)
        assert not con.hdfs.exists(tmp_db_hdfs_path)

    if create_tarball:
        check_call('tar -zc {0} > {1}'
                   .format(IBIS_TEST_DATA_LOCAL_DIR, TARBALL_NAME),
                   shell=True)

    if push_to_s3:
        import boto
        s3_conn = boto.connect_s3(IBIS_TEST_AWS_KEY_ID,
                                  IBIS_TEST_AWS_SECRET)
        bucket = s3_conn.get_bucket(IBIS_TEST_DATA_S3_BUCKET)
        # copy_tarball_to_versioned_backup(bucket)
        key = bucket.new_key(IBIS_TEST_DATA_TARBALL)
        print('Upload tarball to S3')
        key.set_contents_from_filename(TARBALL_NAME, replace=True)


@main.command()
@option('--data/--no-data', default=True, help='Load (skip) ibis testing data')
@option('--udf/--no-udf', default=True, help='Build/upload (skip) test UDFs')
@option('--data-dir',
        help='Path to testing data; dnloads data from S3 if unset')
@option('--overwrite', is_flag=True, help='Forces overwriting of data/UDFs')
def load(data, udf, data_dir, overwrite):
    """Load Ibis test data and build/upload UDFs"""
    print(str(ENV))

    con = make_ibis_client()

    # validate our environment before performing possibly expensive operations
    if not can_write_to_hdfs(con):
        raise IbisError('Failed to write to HDFS; check your settings')
    if udf and not can_build_udfs():
        raise IbisError('Build environment does not support building UDFs')

    # load the data files
    if data:
        tmp_dir = tempfile.mkdtemp(prefix='__ibis_tmp_')

        if not data_dir:
            # TODO(wesm): do not download if already downloaded
            print('Did not specify a local dir with the test data, so '
                  'downloading it from S3')
            data_dir = dnload_ibis_test_data_from_s3(tmp_dir)
        try:
            load_impala_data(con, data_dir, overwrite)

            # sqlite database
            print('Setting up SQLite')
            sqlite_src = osp.join(data_dir, 'ibis_testing.db')
            shutil.copy(sqlite_src, '.')

            print('Loading SQL engines')
            # SQL engines
            engines = [get_postgres_engine()]
            load_sql_databases(con, engines)
        finally:
            shutil.rmtree(tmp_dir)
    else:
        print('Skipping Ibis test data load (--no-data)')

    # build and upload the UDFs
    if udf:
        already_loaded = is_udf_loaded(con)
        print('Attempting to build and load test UDFs')
        if already_loaded and not overwrite:
            print('UDFs already loaded and not overwriting; moving on')
        else:
            if already_loaded:
                print('UDFs already loaded; attempting to overwrite')
            print('Building UDFs')
            build_udfs()
            print('Uploading UDFs')
            upload_udfs(con)
    else:
        print('Skipping UDF build/load (--no-udf)')


def load_impala_data(con, data_dir, overwrite=False):
    already_loaded = is_impala_loaded(con)
    print('Attempting to load Ibis Impala test data (--data)')
    if already_loaded and not overwrite:
        print('Data is already loaded and not overwriting; moving on')
    else:
        if already_loaded:
            print('Data is already loaded; attempting to overwrite')

        print('Uploading to HDFS')
        upload_ibis_test_data_to_hdfs(con, data_dir)
        print('Creating Ibis test data database')
        create_test_database(con)
        parquet_tables = create_parquet_tables(con)
        avro_tables = create_avro_tables(con)
        for table in parquet_tables + avro_tables:
            print('Computing stats for {0}'.format(table.op().name))
            table.compute_stats()


@main.command()
@option('--test-data', is_flag=True,
        help='Cleanup Ibis test data, test database, and also the test UDFs '
             'if they are stored in the test data directory/database')
@option('--udfs', is_flag=True, help='Cleanup Ibis test UDFs only')
@option('--tmp-data', is_flag=True,
        help='Cleanup Ibis temporary HDFS directory')
@option('--tmp-db', is_flag=True, help='Cleanup Ibis temporary database')
def cleanup(test_data, udfs, tmp_data, tmp_db):
    """Cleanup Ibis test data and UDFs"""
    print(str(ENV))

    con = make_ibis_client()

    if udfs:
        # this comes before test_data bc the latter clobbers this too
        con.hdfs.rmdir(pjoin(ENV.test_data_dir, 'udf'))

    if test_data:
        con.drop_database(ENV.test_data_db, force=True)
        con.hdfs.rmdir(ENV.test_data_dir)

    if tmp_data:
        con.hdfs.rmdir(ENV.tmp_dir)

    if tmp_db:
        con.drop_database(ENV.tmp_db, force=True)


if __name__ == '__main__':
    main()
