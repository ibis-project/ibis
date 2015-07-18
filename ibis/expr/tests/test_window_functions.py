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

import ibis

from ibis.compat import unittest
from ibis.expr.tests.mocks import BasicTestCase

from ibis.tests.util import assert_equal


class TestWindowFunctions(BasicTestCase, unittest.TestCase):

    def setUp(self):
        BasicTestCase.setUp(self)
        self.t = self.con.table('alltypes')

    def test_compose_group_by_apis(self):
        t = self.t
        w = ibis.window(group_by=t.g, order_by=t.f)

        diff = t.d - t.d.lag()
        grouped = t.group_by('g').order_by('f')

        expr = grouped[t, diff.name('diff')]
        expr2 = grouped.mutate(diff=diff)
        expr3 = grouped.mutate([diff.name('diff')])

        window_expr = (t.d - t.d.lag().over(w)).name('diff')
        expected = t.projection([t, window_expr])

        assert_equal(expr, expected)
        assert_equal(expr, expr2)
        assert_equal(expr, expr3)

    def test_combine_windows(self):
        pass

    def test_window_bind_to_table(self):
        w = ibis.window(group_by='g', order_by=ibis.desc('f'))

        w2 = w.bind(self.t)
        expected = ibis.window(group_by=self.t.g,
                               order_by=ibis.desc(self.t.f))

        assert_equal(w2, expected)

    def test_window_equals(self):
        pass

    def test_preceding_following_validate(self):
        pass
