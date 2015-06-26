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

from posixpath import join as pjoin
import os
import posixpath
import shutil
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
                             protocol=ENV.impala_protocol)
    hdfs = ibis.hdfs_connect(host=ENV.nn_host, port=ENV.webhdfs_port)
    return ibis.make_client(ic, hdfs_client=hdfs)


def get_ibis_test_data(local_path):
    cmd = 'cd {0} && wget {1} && tar -xzf {2}'.format(
        local_path, IBIS_TEST_DATA_URL, os.path.basename(IBIS_TEST_DATA_URL))
    subprocess.check_call(cmd, shell=True)
    data_dir = pjoin(local_path,
                     os.path.basename(IBIS_TEST_DATA_URL).split('.', 2)[0])
    print('Downloaded {0} and unpacked it to {1}'.format(IBIS_TEST_DATA_URL,
                                                         data_dir))
    return data_dir


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
    for path in parquet_files:
        head, table_name = posixpath.split(path)
        print 'Creating {0}'.format(table_name)
        # if no schema infer!
        schema = schemas.get(table_name)
        con.parquet_file(path, schema=schema, name=table_name,
                         database=ENV.test_data_db, persist=True)


def setup_test_data():
    con = make_connection()
    # TODO: test that HDFS dir is writable before initiating dnload
    try:
        tmp_dir = tempfile.mkdtemp(prefix='__ibis_tmp')
        local_data_dir = get_ibis_test_data(tmp_dir)
        con.hdfs.put(ENV.test_data_dir, local_data_dir, overwrite=True,
                     verbose=True)
    finally:
        shutil.rmtree(tmp_dir)
    create_test_database(con)
    create_parquet_tables(con)


if __name__ == '__main__':
    setup_test_data()
