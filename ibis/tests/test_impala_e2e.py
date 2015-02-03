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

import os
import unittest


import ibis.connection as cnx
import ibis.expr.base as ir


def setup_module():
    try:
        connect(ENV)
    except Exception as e:
        if 'could not connect' in e.message.lower():
            raise unittest.SkipTest


class IbisTestEnv(object):

    def __init__(self):
        self.host = os.environ.get('IBIS_TEST_HOST', 'localhost')
        self.protocol = os.environ.get('IBIS_TEST_PROTOCOL', 'hiveserver2')
        self.port = os.environ.get('IBIS_TEST_PORT', 21050)


ENV = IbisTestEnv()


def connect(env):
    return cnx.impala_connect(host=ENV.host, protocol=ENV.protocol,
                              port=ENV.port)


class TestImpalaConnection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.con = connect(ENV)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_get_table_ref(self):
        table = self.con.table('functional.alltypes')
        assert isinstance(table, ir.TableExpr)

    def test_run_sql(self):
        query = """SELECT li.*
FROM tpch.lineitem li
  INNER JOIN tpch.orders o
    ON li.l_orderkey = o.o_orderkey
"""
        table = self.con.sql(query)

        li = self.con.table('tpch.lineitem')
        assert isinstance(table, ir.TableExpr)
        assert table.schema().equals(li.schema())

        expr = table.limit(10)
        self.con.execute(expr)

    def test_execute_expr(self):
        table = self.con.table('functional.alltypes')
        expr = table.limit(10)
        results = self.con.execute(expr)
        assert len(results) == 10
