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
import os
import posixpath
import shutil

from ibis.util import guid
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

# ----------------------------------------------------------------------
# Functions for creating the test data archive to begin with

TMP_DB_LOCATION = '/__ibis/{0}'.format(guid())
TMP_DB = guid()

def make_temp_database(con):
    if con.exists_database(TMP_DB):
        con.drop_database(TMP_DB, drop_tables=True)
    con.create_database(TMP_DB, path=TMP_DB_LOCATION)
    print('Created database {0} at {1}'.format(TMP_DB, TMP_DB_LOCATION))


def cleanup_temporary_stuff(con):
    con.drop_database(TMP_DB, drop_tables=True)
    assert not con.hdfs.exists(TMP_DB_LOCATION)

def download_parquet_files(con):
    parquet_path = pjoin(TEST_DATA_DIR, 'parquet')
    print("Downloading {0}".format(parquet_path))
    con.hdfs.get(TMP_DB_LOCATION, parquet_path)


def download_avro_files(con):
    avro_path = '/test-warehouse/tpch.region_avro'
    os.mkdir(os.path.join(TEST_DATA_DIR, 'avro'))
    print("Downloading {0}".format(avro_path))
    con.hdfs.get(avro_path, pjoin(TEST_DATA_DIR, 'avro', 'tpch.region'))


def generate_csv_files():
    import numpy as np
    import pandas as pd
    import pandas.util.testing as tm

    N = 10
    nfiles = 10

    csv_base = os.path.join(TEST_DATA_DIR, 'csv')
    os.mkdir(csv_base)

    df = pd.DataFrame({
        'foo': [tm.rands(10) for _ in xrange(N)],
        'bar': np.random.randn(N),
        'baz': np.random.randint(0, 100, size=N)
    }, columns=['foo', 'bar', 'baz'])

    for i in xrange(nfiles):
        csv_path = os.path.join(csv_base, '{}.csv'.format(i))
        print('Writing {}'.format(csv_path))
        df.to_csv(csv_path, index=False, header=False)


def scrape_parquet_files(con):
    to_scrape = [('tpch', x) for x in con.list_tables(database='tpch')]
    to_scrape.append(('functional', 'alltypes'))
    for db, tname in to_scrape:
        table = con.table(tname, database=db)
        new_name = '{}_{}'.format(db, tname)
        print('Creating {}'.format(new_name))
        con.create_table(new_name, table, database=TMP_DB)


def make_local_test_archive():
    con = make_connection()
    make_temp_database(con)

    try:
        scrape_parquet_files(con)

        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR)

        download_parquet_files(con)
        download_avro_files(con)
        generate_csv_files()
    finally:
        cleanup_temporary_stuff(con)

# ----------------------------------------------------------------------
#


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


def setup_test_data():
    con = make_connection()
    write_data_to_hdfs(con)
    create_test_database(con)
    create_parquet_tables(con)


if __name__ == '__main__':
    setup_test_data()
