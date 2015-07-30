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

import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis

from ibis.compat import unittest
from ibis.expr.tests.mocks import BasicTestCase
from ibis.tests.util import assert_equal


class TestCaseExpressions(BasicTestCase, unittest.TestCase):

    def test_ifelse(self):
        bools = self.table.g.isnull()
        result = bools.ifelse("foo", "bar")
        assert isinstance(result, ir.StringArray)

    def test_ifelse_literal(self):
        pass

    def test_simple_case_expr(self):
        case1, result1 = "foo", self.table.a
        case2, result2 = "bar", self.table.c
        default_result = self.table.b

        expr1 = self.table.g.lower().cases(
            [(case1, result1),
             (case2, result2)],
            default=default_result
        )

        expr2 = (self.table.g.lower().case()
                 .when(case1, result1)
                 .when(case2, result2)
                 .else_(default_result)
                 .end())

        assert_equal(expr1, expr2)
        assert isinstance(expr1, ir.Int32Array)

    def test_multiple_case_expr(self):
        case1 = self.table.a == 5
        case2 = self.table.b == 128
        case3 = self.table.c == 1000

        result1 = self.table.f
        result2 = self.table.b * 2
        result3 = self.table.e

        default = self.table.d

        expr = (ibis.case()
                .when(case1, result1)
                .when(case2, result2)
                .when(case3, result3)
                .else_(default)
                .end())

        op = expr.op()
        assert isinstance(expr, ir.DoubleArray)
        assert isinstance(op, ops.SearchedCase)
        assert op.default is default

    def test_simple_case_no_default(self):
        # TODO: this conflicts with the null else cases below. Make a decision
        # about what to do, what to make the default behavior based on what the
        # user provides. SQL behavior is to use NULL when nothing else
        # provided. The .replace convenience API could use the field values as
        # the default, getting us around this issue.
        pass

    def test_simple_case_null_else(self):
        expr = self.table.g.case().when("foo", "bar").end()
        op = expr.op()

        assert isinstance(expr, ir.StringArray)
        assert isinstance(op.default, ir.ValueExpr)
        assert isinstance(op.default.op(), ir.NullLiteral)

    def test_multiple_case_null_else(self):
        expr = ibis.case().when(self.table.g == "foo", "bar").end()
        op = expr.op()

        assert isinstance(expr, ir.StringArray)
        assert isinstance(op.default, ir.ValueExpr)
        assert isinstance(op.default.op(), ir.NullLiteral)

    def test_case_type_precedence(self):
        pass

    def test_no_implicit_cast_possible(self):
        pass
