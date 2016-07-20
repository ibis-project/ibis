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

from ibis import literal
import ibis.expr.types as ir
import ibis.expr.operations as ops

from ibis.expr.tests.mocks import MockConnection
from ibis.compat import unittest
from ibis.tests.util import assert_equal


class TestStringOps(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_lower_upper(self):
        lresult = self.table.g.lower()
        uresult = self.table.g.upper()

        assert isinstance(lresult, ir.StringArray)
        assert isinstance(uresult, ir.StringArray)

        assert isinstance(lresult.op(), ops.Lowercase)
        assert isinstance(uresult.op(), ops.Uppercase)

        lit = literal('FoO')

        lresult = lit.lower()
        uresult = lit.upper()
        assert isinstance(lresult, ir.StringScalar)
        assert isinstance(uresult, ir.StringScalar)

    def test_substr(self):
        lit = literal('FoO')

        result = self.table.g.substr(2, 4)
        lit_result = lit.substr(0, 2)

        assert isinstance(result, ir.StringArray)
        assert isinstance(lit_result, ir.StringScalar)

        op = result.op()
        assert isinstance(op, ops.Substring)

        start, length = op.args[1:]

        assert start.equals(literal(2))
        assert length.equals(literal(4))

    def test_left_right(self):
        result = self.table.g.left(5)
        expected = self.table.g.substr(0, 5)
        assert result.equals(expected)

        result = self.table.g.right(5)
        op = result.op()
        assert isinstance(op, ops.StrRight)
        assert op.args[1].equals(literal(5))

    def test_length(self):
        lit = literal('FoO')
        result = self.table.g.length()
        lit_result = lit.length()

        assert isinstance(result, ir.Int32Array)
        assert isinstance(lit_result, ir.Int32Scalar)
        assert isinstance(result.op(), ops.StringLength)

    def test_join(self):
        dash = literal('-')

        expr = dash.join([self.table.f.cast('string'),
                          self.table.g])
        assert isinstance(expr, ir.StringArray)

        expr = dash.join([literal('ab'), literal('cd')])
        assert isinstance(expr, ir.StringScalar)

    def test_contains(self):
        expr = self.table.g.contains('foo')
        expected = self.table.g.find('foo') >= 0
        assert_equal(expr, expected)

        self.assertRaises(Exception, lambda: 'foo' in self.table.g)

    def test_getitem_slice(self):
        cases = [
            (self.table.g[:3], self.table.g.substr(0, 3)),
            (self.table.g[2:6], self.table.g.substr(2, 4)),
        ]

        for case, expected in cases:
            assert_equal(case, expected)
