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

# Populates the ibis_testing Impala database

from posixpath import join as pjoin
import posixpath

import ibis

IMPALA_HOST = 'localhost'
HDFS_HOST = 'localhost'
WEBHDFS_PORT = 5070
TEST_DB = 'ibis_testing'
TEST_DATA_DIR = 'ibis-testing-data'
TEST_DATA_HDFS_LOC = '/__ibis/ibis-testing-data'


def make_connection():
    ic = ibis.impala_connect(host=IMPALA_HOST)
    hdfs = ibis.hdfs_connect(host=HDFS_HOST, port=WEBHDFS_PORT)
    con = ibis.make_client(ic, hdfs_client=hdfs)

    return con


def write_data_to_hdfs(con):
    # TODO per #278, write directly from the gzipped tarball
    con.hdfs.put(TEST_DATA_HDFS_LOC, TEST_DATA_DIR,
                 verbose=True, overwrite=True)


def create_test_database(con):
    if con.exists_database(TEST_DB):
        con.drop_database(TEST_DB, drop_tables=True)
    con.create_database(TEST_DB)
    print('Created database {0}'.format(TEST_DB))


def create_parquet_tables(con):
    parquet_files = con.hdfs.ls(pjoin(TEST_DATA_HDFS_LOC, 'parquet'))

    for path in parquet_files:
        head, table_name = posixpath.split(path)
        print 'Creating {0}'.format(table_name)
        con.parquet_file(path, name=table_name, database=TEST_DB, persist=True)


if __name__ == '__main__':
    con = make_connection()
    write_data_to_hdfs(con)
    create_test_database(con)
    create_parquet_tables(con)
