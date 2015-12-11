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
import time
import six

import pytest

from ibis import options
import ibis.util as util
import ibis.compat as compat
import ibis


GLOBAL_TMP_DB = os.environ.get('IBIS_TEST_TMP_DB',
                               '__ibis_tmp_{0}'.format(util.guid()))


GLOBAL_TMP_DIR = os.environ.get('IBIS_TEST_TMP_HDFS_DIR',
                                '/tmp/__ibis_test')

# update global Ibis config where relevant
options.impala.temp_db = GLOBAL_TMP_DB
options.impala.temp_hdfs_path = GLOBAL_TMP_DIR


class IbisTestEnv(object):

    def __init__(self):
        # TODO: allow initializing values through a constructor
        self.impala_host = os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')
        self.impala_port = int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))
        self.tmp_db = GLOBAL_TMP_DB
        self.tmp_dir = GLOBAL_TMP_DIR
        self.test_data_db = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')
        self.test_data_dir = os.environ.get('IBIS_TEST_DATA_HDFS_DIR',
                                            '/__ibis/ibis-testing-data')
        self.nn_host = os.environ.get('IBIS_TEST_NN_HOST', 'localhost')
        # 5070 is default for impala dev env
        self.webhdfs_port = int(os.environ.get('IBIS_TEST_WEBHDFS_PORT', 5070))
        self.hdfs_superuser = os.environ.get('IBIS_TEST_HDFS_SUPERUSER',
                                             'hdfs')
        self.use_codegen = os.environ.get('IBIS_TEST_USE_CODEGEN',
                                          'False').lower() == 'true'
        self.cleanup_test_data = os.environ.get('IBIS_TEST_CLEANUP_TEST_DATA',
                                                'True').lower() == 'true'
        self.auth_mechanism = os.environ.get('IBIS_TEST_AUTH_MECH', 'NOSASL')
        self.llvm_config = os.environ.get('IBIS_TEST_LLVM_CONFIG', None)

    def __repr__(self):
        kvs = ['{0}={1}'.format(k, v)
               for (k, v) in six.iteritems(self.__dict__)]
        return 'IbisTestEnv(\n    {0})'.format(',\n    '.join(kvs))


ENV = IbisTestEnv()


def connect_test(env, with_hdfs=True):
    if with_hdfs:
        if env.auth_mechanism in ['GSSAPI', 'LDAP']:
            print("Warning: ignoring invalid Certificate Authority errors")
        hdfs_client = ibis.hdfs_connect(host=env.nn_host,
                                        port=env.webhdfs_port,
                                        auth_mechanism=env.auth_mechanism,
                                        verify=(env.auth_mechanism
                                                not in ['GSSAPI', 'LDAP']))
    else:
        hdfs_client = None

    return ibis.impala.connect(host=env.impala_host,
                               database=env.test_data_db,
                               port=env.impala_port,
                               auth_mechanism=env.auth_mechanism,
                               pool_size=2,
                               hdfs_client=hdfs_client)


@pytest.mark.impala
class ImpalaE2E(object):

    @classmethod
    def setUpClass(cls):
        ImpalaE2E.setup_e2e(cls, ENV)

        # make sure this never gets messed up
        opts = cls.con.get_options()
        assert opts['DISABLE_CODEGEN'] == '1'

    @classmethod
    def tearDownClass(cls):
        ImpalaE2E.teardown_e2e(cls)

    @staticmethod
    def setup_e2e(cls, env):
        cls.env = env
        cls.con = connect_test(env)

        # Tests run generally faster without it
        if not env.use_codegen:
            cls.con.disable_codegen()
        cls.hdfs = cls.con.hdfs
        cls.test_data_dir = env.test_data_dir
        cls.test_data_db = env.test_data_db
        cls.tmp_dir = env.tmp_dir
        cls.tmp_db = env.tmp_db

        try:
            cls.alltypes = cls.con.table('functional_alltypes')
        except:
            pass

        cls.db = cls.con.database(env.test_data_db)

        if not cls.con.exists_database(cls.tmp_db):
            cls.con.create_database(cls.tmp_db)

        if not cls.hdfs.exists(cls.tmp_dir):
            cls.hdfs.mkdir(cls.tmp_dir)
        cls.hdfs.chmod(cls.tmp_dir, '777')

    @staticmethod
    def teardown_e2e(cls):
        i, retries = 0, 3
        while True:
            # reduce test flakiness
            try:
                cls.con.drop_database(cls.tmp_db, force=True)
                break
            except:
                i += 1
                if i >= retries:
                    raise

                time.sleep(0.1)

    def setUp(self):
        self.temp_databases = []
        self.temp_tables = []
        self.temp_views = []
        self.temp_udfs = []
        self.temp_udas = []

    def tearDown(self):
        for t in self.temp_tables:
            self.con.drop_table(t, force=True)

        for t in self.temp_views:
            self.con.drop_view(t, force=True)

        for f_name, f_inputs in self.temp_udfs:
            self.con.drop_udf(f_name, input_types=f_inputs, force=True)

        for f_name, f_inputs in self.temp_udas:
            self.con.drop_uda(f_name, input_types=f_inputs, force=True)

        self.con.set_database(self.test_data_db)
        for t in self.temp_databases:
            self.con.drop_database(t, force=True)


def format_schema(expr):
    from ibis.impala.compiler import _type_to_sql_string
    from pprint import pprint
    schema = expr.schema()

    what = compat.lzip(schema.names,
                       [_type_to_sql_string(x) for x in schema.types])
    pprint(what)
