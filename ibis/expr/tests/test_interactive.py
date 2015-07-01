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

from ibis.compat import unittest
from ibis.expr.tests.mocks import MockConnection
import ibis.config as config

from ibis.tests.util import assert_equal


class TestInteractiveUse(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()

    def test_interactive_execute_on_repr(self):
        table = self.con.table('functional_alltypes')
        expr = table.bigint_col.sum()
        with config.option_context('interactive', True):
            repr(expr)

        assert self.con.last_executed_expr is expr

    def test_default_limit(self):
        table = self.con.table('functional_alltypes')

        with config.option_context('interactive', True):
            repr(table)

        result = self.con.last_executed_expr
        limit = config.options.sql.default_limit
        assert_equal(result, table.limit(limit))

    def test_disable_query_limit(self):
        table = self.con.table('functional_alltypes')

        with config.option_context('interactive', True):
            with config.option_context('sql.default_limit', None):
                repr(table)

        result = self.con.last_executed_expr
        assert_equal(result, table)

    def test_interactive_non_compilable_repr_not_fail(self):
        # #170
        table = self.con.table('functional_alltypes')

        expr = table.string_col.topk(3)

        # it works!
        with config.option_context('interactive', True):
            repr(expr)

    def test_histogram_repr_no_query_execute(self):
        t = self.con.table('functional_alltypes')
        tier = t.double_col.histogram(10).name('bucket')
        expr = t.group_by(tier).size()
        with config.option_context('interactive', True):
            expr._repr()
        assert self.con.last_executed_expr is None
