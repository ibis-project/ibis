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

from ibis.compat import unittest
import ibis.expr.types as ir

import ibis.sql.sqlite.api as api


@pytest.mark.sqlite
class SQLiteTests(object):
    pass


class SQLiteTestEnv(object):

    def __init__(self):
        self.db_path = os.environ.get('IBIS_TEST_SQLITE_DB_PATH',
                                      'ibis_testing.db')


class TestSQLiteClient(unittest.TestCase, SQLiteTests):

    @classmethod
    def setUpClass(cls):
        cls.env = SQLiteTestEnv()
        cls.db = api.connect(cls.env.db_path)

        cls.alltypes = cls.db.table('functional_alltypes')

    @classmethod
    def tearDownClass(cls):
        pass

    def test_table(self):
        table = self.db.table('functional_alltypes')
        assert isinstance(table, ir.TableExpr)

    def test_simple_aggregate_execute(self):
        d = self.alltypes.double_col
        # d.execute()
