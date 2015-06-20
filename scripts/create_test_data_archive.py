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

# Populates the ibis_testing Impala database

from posixpath import join as pjoin
import os
import posixpath
import shutil
import tempfile
import subprocess

import numpy as np
import pandas as pd
import pandas.util.testing as tm

from ibis.util import guid
from ibis.tests.util import IbisTestEnv
import ibis


ENV = IbisTestEnv()
TMP_DB_HDFS_PATH = pjoin(ENV.tmp_dir, guid())
TMP_DB = guid()
# hardcoded:
IBIS_TEST_DATA_LOCAL_DIR = 'ibis-testing-data'


def make_connection():
    ic = ibis.impala_connect(host=ENV.impala_host, port=ENV.impala_port,
                             protocol=ENV.impala_protocol)
    hdfs = ibis.hdfs_connect(host=ENV.nn_host, port=ENV.webhdfs_port)
    return ibis.make_client(ic, hdfs_client=hdfs)


def make_temp_database(con):
    if con.exists_database(TMP_DB):
        con.drop_database(TMP_DB, force=True)
    con.create_database(TMP_DB, path=TMP_DB_HDFS_PATH)
    print('Created database {0} at {1}'.format(TMP_DB, TMP_DB_HDFS_PATH))


def scrape_parquet_files(con):
    to_scrape = [('tpch', x) for x in con.list_tables(database='tpch')]
    to_scrape.append(('functional', 'alltypes'))
    for db, tname in to_scrape:
        table = con.table(tname, database=db)
        new_name = '{0}_{1}'.format(db, tname)
        print('Creating {0}'.format(new_name))
        con.create_table(new_name, table, database=TMP_DB)


def download_parquet_files(con):
    parquet_path = pjoin(IBIS_TEST_DATA_LOCAL_DIR, 'parquet')
    print("Downloading {0}".format(parquet_path))
    con.hdfs.get(TMP_DB_HDFS_PATH, parquet_path)


def download_avro_files(con):
    avro_path = '/test-warehouse/tpch.region_avro'
    os.mkdir(os.path.join(IBIS_TEST_DATA_LOCAL_DIR, 'avro'))
    print("Downloading {0}".format(avro_path))
    con.hdfs.get(avro_path,
                 pjoin(IBIS_TEST_DATA_LOCAL_DIR, 'avro', 'tpch.region'))


def generate_csv_files():
    N = 10
    nfiles = 10

    csv_base = os.path.join(IBIS_TEST_DATA_LOCAL_DIR, 'csv')
    os.mkdir(csv_base)

    df = pd.DataFrame({
        'foo': [tm.rands(10) for _ in xrange(N)],
        'bar': np.random.randn(N),
        'baz': np.random.randint(0, 100, size=N)
    }, columns=['foo', 'bar', 'baz'])

    for i in xrange(nfiles):
        csv_path = os.path.join(csv_base, '{0}.csv'.format(i))
        print('Writing {0}'.format(csv_path))
        df.to_csv(csv_path, index=False, header=False)


def cleanup_temporary_stuff(con):
    con.drop_database(TMP_DB, force=True)
    assert not con.hdfs.exists(TMP_DB_HDFS_PATH)


def make_local_test_archive():
    con = make_connection()
    make_temp_database(con)

    try:
        scrape_parquet_files(con)

        if os.path.exists(IBIS_TEST_DATA_LOCAL_DIR):
            shutil.rmtree(IBIS_TEST_DATA_LOCAL_DIR)
        os.mkdir(IBIS_TEST_DATA_LOCAL_DIR)

        download_parquet_files(con)
        download_avro_files(con)
        generate_csv_files()
    finally:
        cleanup_temporary_stuff(con)

    # TODO: push a tarball to S3?


if __name__ == '__main__':
    make_local_test_archive()
