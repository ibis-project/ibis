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

from .common import PostgreSQLTests
from ibis.compat import unittest
from ibis.tests.util import assert_equal
import ibis.expr.types as ir
import ibis


class TestPostgreSQLClient(PostgreSQLTests, unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        pass

    def test_table(self):
        table = self.con.table('functional_alltypes')
        assert isinstance(table, ir.TableExpr)

    def test_array_execute(self):
        d = self.alltypes.limit(10).double_col
        s = d.execute()
        assert isinstance(s, pd.Series)
        assert len(s) == 10

    def test_literal_execute(self):
        expr = ibis.literal('1234')
        result = self.con.execute(expr)
        assert result == '1234'

    def test_simple_aggregate_execute(self):
        d = self.alltypes.double_col.sum()
        v = d.execute()
        assert isinstance(v, float)

    def test_list_tables(self):
        assert len(self.con.list_tables()) > 0
        assert len(self.con.list_tables(like='functional')) == 1

    def test_compile_verify(self):
        unsupported_expr = self.alltypes.string_col.approx_nunique()
        assert not unsupported_expr.verify()

        supported_expr = self.alltypes.double_col.sum()
        assert supported_expr.verify()

    def test_database_layer(self):
        db = self.con.database()

        t = db.functional_alltypes
        assert_equal(t, self.alltypes)

        assert db.list_tables() == self.con.list_tables()

    def test_compile_toplevel(self):
        # t = ibis.table([
        #     ('foo', 'double')
        # ])

        # # it works!
        # expr = t.foo.sum()
        # ibis.postgres.compile(expr)

        # This does not work yet because if the compiler encounters a
        # non-SQLAlchemy table it fails
        pass
