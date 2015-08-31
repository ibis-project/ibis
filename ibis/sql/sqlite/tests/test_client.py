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

import pandas as pd

from .common import SQLiteTests, SQLiteTestEnv
from ibis.compat import unittest
import ibis.expr.types as ir

import ibis.sql.sqlite.api as api


class TestSQLiteClient(SQLiteTests, unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        pass

    def test_table(self):
        table = self.db.table('functional_alltypes')
        assert isinstance(table, ir.TableExpr)

    def test_array_execute(self):
        d = self.alltypes.limit(10).double_col
        s = d.execute()
        assert isinstance(s, pd.Series)
        assert len(s) == 10

    def test_simple_aggregate_execute(self):
        d = self.alltypes.double_col.sum()
        v = d.execute()
        assert isinstance(v, float)

    def test_list_tables(self):
        assert len(self.db.list_tables()) > 0
        assert len(self.db.list_tables(like='functional')) == 1
