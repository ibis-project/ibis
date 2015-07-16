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

import os

import pytest

from ibis import Schema
from ibis import options
import ibis.util as util
import ibis


class IbisTestEnv(object):

    def __init__(self):
        # TODO: allow initializing values through a constructor
        self.impala_host = os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')
        self.impala_protocol = os.environ.get('IBIS_TEST_IMPALA_PROTOCOL',
                                              'hiveserver2')
        self.impala_port = int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))
        self.tmp_db = os.environ.get('IBIS_TEST_TMP_DB', '__ibis_tmp')
        self.tmp_dir = os.environ.get('IBIS_TEST_TMP_HDFS_DIR', '/tmp/__ibis_test')
        self.test_data_db = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')
        self.test_data_dir = os.environ.get('IBIS_TEST_DATA_HDFS_DIR',
                                            '/__ibis/ibis-testing-data')
        self.nn_host = os.environ.get('IBIS_TEST_NN_HOST', 'localhost')
        # 5070 is default for impala dev env
        self.webhdfs_port = int(os.environ.get('IBIS_TEST_WEBHDFS_PORT', 5070))
        self.hdfs_url = 'http://{0}:{1}'.format(self.nn_host,
                                                self.webhdfs_port)
        self.use_codegen = os.environ.get('IBIS_TEST_USE_CODEGEN',
                                          'False').lower() == 'true'
        self.cleanup_test_data = os.environ.get('IBIS_TEST_CLEANUP_TEST_DATA',
                                                'True').lower() == 'true'
        self.use_kerberos = os.environ.get('IBIS_TEST_USE_KERBEROS',
                                           'False').lower() == 'true'

        # update global Ibis config where relevant
        options.impala.temp_db = self.tmp_db
        options.impala.temp_hdfs_path = self.tmp_dir

    def __repr__(self):
        kvs = ['{0}={1}'.format(k, v) for (k, v) in self.__dict__.iteritems()]
        return 'IbisTestEnv(\n    {0})'.format(',\n    '.join(kvs))


def test_connect(env, with_hdfs=True):
    con = ibis.impala_connect(host=env.impala_host,
                              protocol=env.impala_protocol,
                              database=env.test_data_db,
                              port=env.impala_port,
                              use_kerberos=env.use_kerberos)
    if with_hdfs:
        if env.use_kerberos:
            from hdfs.ext.kerberos import KerberosClient
            hdfs_client = KerberosClient(env.hdfs_url, mutual_auth='REQUIRED')
        else:
            from hdfs.client import InsecureClient
            hdfs_client = InsecureClient(env.hdfs_url)
        return ibis.make_client(con, hdfs_client)
    else:
        return ibis.make_client(con)


@pytest.mark.e2e
class ImpalaE2E(object):

    @classmethod
    def setUpClass(cls):
        ENV = IbisTestEnv()
        cls.con = test_connect(ENV)
        # Tests run generally faster without it
        if not ENV.use_codegen:
            cls.con.disable_codegen()
        cls.hdfs = cls.con.hdfs
        cls.test_data_dir = ENV.test_data_dir
        cls.test_data_db = ENV.test_data_db
        cls.tmp_dir = ENV.tmp_dir
        cls.tmp_db = ENV.tmp_db
        cls.alltypes = cls.con.table('functional_alltypes')

        if not cls.con.exists_database(cls.tmp_db):
            cls.con.create_database(cls.tmp_db)

    @classmethod
    def tearDownClass(cls):
        cls.con.drop_database(cls.tmp_db, force=True)

    def setUp(self):
        self.temp_databases = []
        self.temp_tables = []
        self.temp_views = []

    def tearDown(self):
        for t in self.temp_tables:
            self.con.drop_table(t, force=True)

        for t in self.temp_views:
            self.con.drop_view(t, force=True)

        self.con.set_database(self.test_data_db)
        for t in self.temp_databases:
            self.con.drop_database(t, force=True)



def assert_equal(left, right):
    if util.all_of([left, right], Schema):
        assert left.equals(right),\
            'Comparing schemas: \n%s !=\n%s' % (repr(left), repr(right))
    else:
        assert left.equals(right), ('Objects unequal: {0}\nvs\n{1}'
                                    .format(repr(left), repr(right)))
