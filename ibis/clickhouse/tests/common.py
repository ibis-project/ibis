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

# update global Ibis config where relevant
options.clickhouse.temp_db = GLOBAL_TMP_DB


class IbisTestEnv(object):

    def __init__(self):
        # TODO: allow initializing values through a constructor
        self.clickhouse_host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST',
                                              'localhost')
        self.clickhouse_port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT',
                                                  9000))
        self.tmp_db = GLOBAL_TMP_DB
        self.test_data_db = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')

        self.cleanup_test_data = os.environ.get('IBIS_TEST_CLEANUP_TEST_DATA',
                                                'True').lower() == 'true'

    def __repr__(self):
        kvs = ['{0}={1}'.format(k, v)
               for (k, v) in six.iteritems(self.__dict__)]
        return 'IbisTestEnv(\n    {0})'.format(',\n    '.join(kvs))


ENV = IbisTestEnv()


def connect_test(env):
    return ibis.clickhouse.connect(host=env.clickhouse_host,
                                   database=env.test_data_db,
                                   port=env.clickhouse_port)


@pytest.mark.clickhouse
class ClickhouseE2E(object):

    @classmethod
    def setUpClass(cls):
        ClickhouseE2E.setup_e2e(cls, ENV)

    @classmethod
    def tearDownClass(cls):
        ClickhouseE2E.teardown_e2e(cls)

    @staticmethod
    def setup_e2e(cls, env):
        cls.env = env
        cls.con = connect_test(env)

        # Tests run generally faster without it
        cls.test_data_db = env.test_data_db
        cls.tmp_db = env.tmp_db
        cls.alltypes = cls.con.table('functional_alltypes')
        cls.db = cls.con.database(env.test_data_db)

        if not cls.con.exists_database(cls.tmp_db):
            cls.con.create_database(cls.tmp_db)

    @staticmethod
    def teardown_e2e(cls):
        i, retries = 0, 3
        while True:
            # reduce test flakiness
            try:
                cls.con.drop_database(cls.tmp_db, force=True)
            except Exception:
                i += 1
                if i >= retries:
                    raise

                time.sleep(0.1)
            else:
                break

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


def format_schema(expr):
    from ibis.clickhouse.compiler import _type_to_sql_string
    from pprint import pprint
    schema = expr.schema()

    what = compat.lzip(schema.names,
                       [_type_to_sql_string(x) for x in schema.types])
    pprint(what)
