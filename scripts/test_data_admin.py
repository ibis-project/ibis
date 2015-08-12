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

import sys
import shutil
import tempfile
import os.path as osp
from os.path import join as pjoin
from subprocess import check_call

import pandas as pd
from click import group, option

import ibis
from ibis.compat import BytesIO
from ibis.common import IbisError
from ibis.tests.util import IbisTestEnv


ENV = IbisTestEnv()
IBIS_TEST_DATA_S3_BUCKET = 'ibis-test-resources'
IBIS_TEST_DATA_LOCAL_DIR = 'ibis-testing-data'
IBIS_TEST_DATA_TARBALL = 'ibis-testing-data.tar.gz'


def make_ibis_client():
    ic = ibis.impala_connect(host=ENV.impala_host, port=ENV.impala_port,
                             protocol=ENV.impala_protocol,
                             use_kerberos=ENV.use_kerberos)
    if ENV.use_kerberos:
        print("Warning: ignoring invalid Certificate Authority errors")
    hc = ibis.hdfs_connect(host=ENV.nn_host, port=ENV.webhdfs_port,
                           use_kerberos=ENV.use_kerberos,
                           verify=(not ENV.use_kerberos))
    return ibis.make_client(ic, hdfs_client=hc)


def can_write_to_hdfs(con):
    test_path = pjoin(ENV.test_data_dir, ibis.util.guid())
    test_file = BytesIO(ibis.util.guid())
    try:
        con.hdfs.put(test_path, test_file)
        con.hdfs.rm(test_path)
        return True
    except:
        return False


def can_build_udfs():
    try:
        check_call('which cmake', shell=True)
        check_call('which make', shell=True)
        check_call('which clang++', shell=True)
        return True
    except:
        return False


def is_data_loaded(con):
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
        local_path, url, IBIS_TEST_DATA_TARBALL)
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
    for path in parquet_files:
        head, table_name = osp.split(path)
        print('Creating {0}'.format(table_name))
        # if no schema infer!
        schema = schemas.get(table_name)
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
    for path in avro_files:
        head, table_name = osp.split(path)
        print('Creating {0}'.format(table_name))
        schema = schemas[table_name]
        table = con.avro_file(path, schema, name=table_name,
                          database=ENV.test_data_db, persist=True)
        tables.append(table)
    return tables


def build_udfs():
    print('Building UDFs')
    ibis_home_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    udf_dir = pjoin(ibis_home_dir, 'testing', 'udf')
    check_call('cmake . && make', shell=True, cwd=udf_dir)


def upload_udfs(con):
    ibis_home_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    build_dir = pjoin(ibis_home_dir, 'testing', 'udf', 'build')
    bitcode_dir = pjoin(ENV.test_data_dir, 'udf')
    print('Uploading UDFs to {0}'.format(bitcode_dir))
    if con.hdfs.exists(bitcode_dir):
        con.hdfs.rmdir(bitcode_dir)
    con.hdfs.put(bitcode_dir, build_dir, verbose=True)


def scrape_parquet_files(con):
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
        scrape_parquet_files(con)
        download_parquet_files(con, tmp_db_hdfs_path)
        download_avro_files(con)
        generate_csv_files()
    finally:
        con.drop_database(tmp_db, force=True)
        assert not con.hdfs.exists(TMP_DB_HDFS_PATH)

    if create_tarball:
        check_call('tar -xzf {0} {1}'.format(IBIS_TEST_DATA_TARBALL,
                                             IBIS_TEST_DATA_LOCAL_DIR),
                   shell=True)

    if push_to_s3:
        from boto.s3 import connect_to_region
        s3_conn = connect_to_region('us-west-2')
        bucket = s3_conn.get_bucket(IBIS_TEST_DATA_S3_BUCKET)
        copy_tarball_to_versioned_backup(bucket)
        key = bucket.new_key(IBIS_TEST_DATA_TARBALL)
        print('Upload tarball to S3')
        key.set_contents_from_filename(IBIS_TEST_DATA_TARBALL, replace=False)


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
    if data and (overwrite or not is_data_loaded(con)):
        try:
            tmp_dir = tempfile.mkdtemp(prefix='__ibis_tmp_')
            if not data_dir:
                print('Did not specify a local dir with the test data, so '
                      'downloading it from S3')
                data_dir = dnload_ibis_test_data_from_s3(tmp_dir)
            upload_ibis_test_data_to_hdfs(con, data_dir)
            create_test_database(con)
            parquet_tables = create_parquet_tables(con)
            avro_tables = create_avro_tables(con)
            for table in parquet_tables + avro_tables:
                print('Computing stats for {0}'.format(table.op().name))
                table.compute_stats()
        finally:
            shutil.rmtree(tmp_dir)

    # build and upload the UDFs
    if udf and (overwrite or not is_udf_loaded(con)):
        build_udfs()
        upload_udfs(con)


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
