#! /usr/bin/env python
# Copyright 2014 Cloudera Inc.
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

# Fetches the ibis-testing-data archive and loads it into Impala

from __future__ import print_function

from posixpath import join as pjoin
import os
import posixpath
import shutil
import sys
import tempfile
import subprocess

import ibis
from ibis.tests.util import IbisTestEnv


ENV = IbisTestEnv()
# hardcoded:
IBIS_TEST_DATA_URL = ('https://ibis-test-resources.s3.amazonaws.com/'
                      'ibis-testing-data.tar.gz')


def make_connection():
    ic = ibis.impala_connect(host=ENV.impala_host, port=ENV.impala_port,
                             protocol=ENV.impala_protocol,
                             use_kerberos=ENV.use_kerberos)
    if ENV.use_kerberos:
        print("Warning: ignoring invalid Certificate Authority errors")
    hdfs = ibis.hdfs_connect(host=ENV.nn_host, port=ENV.webhdfs_port,
                             use_kerberos=ENV.use_kerberos,
                             verify=(not ENV.use_kerberos))
    return ibis.make_client(ic, hdfs_client=hdfs)


def get_ibis_test_data(local_path):
    cmd = 'cd {0} && wget -q {1} && tar -xzf {2}'.format(
        local_path, IBIS_TEST_DATA_URL, os.path.basename(IBIS_TEST_DATA_URL))
    subprocess.check_call(cmd, shell=True)
    data_dir = pjoin(local_path,
                     os.path.basename(IBIS_TEST_DATA_URL).split('.', 2)[0])
    print('Downloaded {0} and unpacked it to {1}'.format(IBIS_TEST_DATA_URL,
                                                         data_dir))
    return data_dir


def create_udf_data(con):
    os.chdir('../testing/udf')
    subprocess.check_call('cmake .', shell=True)
    subprocess.check_call('make', shell=True)
    build_dir = 'build/'
    so_dir = ENV.test_data_dir + '/udf'
    con.hdfs.put(so_dir, build_dir, verbose=True)


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
        head, table_name = posixpath.split(path)
        print('Creating {0}'.format(table_name))
        # if no schema infer!
        schema = schemas.get(table_name)
        t = con.parquet_file(path, schema=schema, name=table_name,
                             database=ENV.test_data_db, persist=True)
        tables.append(t)

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
        head, table_name = posixpath.split(path)
        print('Creating {0}'.format(table_name))
        schema = schemas[table_name]
        t = con.avro_file(path, schema, name=table_name,
                          database=ENV.test_data_db, persist=True)
        tables.append(t)

    return tables


def setup_test_data(local_data_dir):
    con = make_connection()
    hdfs = con.hdfs

    if hdfs.exists(ENV.test_data_dir):
        hdfs.rmdir(ENV.test_data_dir)
    hdfs.put(ENV.test_data_dir, local_data_dir, verbose=True)

    create_udf_data(con)
    create_test_database(con)
    parquet_tables = create_parquet_tables(con)
    avro_tables = create_avro_tables(con)

    for t in parquet_tables + avro_tables:
        print('Computing stats for {0}'.format(t.op().name))
        t.compute_stats()


def can_write_to_hdfs():
    from ibis.compat import BytesIO
    con = make_connection()

    test_path = pjoin(ENV.test_data_dir, ibis.util.guid())
    test_file = BytesIO(ibis.util.guid())

    try:
        con.hdfs.put(test_path, test_file)
        con.hdfs.rm(test_path)
        return True
    except:
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_dir = os.path.expanduser(sys.argv[1])
        setup_test_data(data_dir)
    else:
        if not can_write_to_hdfs():
            print('Do not have write permission to HDFS')

        try:
            tmp_dir = tempfile.mkdtemp(prefix='__ibis_tmp')
            local_data_dir = get_ibis_test_data(tmp_dir)
            setup_test_data(local_data_dir)
        finally:
            shutil.rmtree(tmp_dir)
