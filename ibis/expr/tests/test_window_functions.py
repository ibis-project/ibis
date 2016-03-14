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
        t = self.t
        w1 = ibis.window(group_by=t.g, order_by=t.f)
        w2 = ibis.window(preceding=5, following=5)

        w3 = w1.combine(w2)
        expected = ibis.window(group_by=t.g, order_by=t.f,
                               preceding=5, following=5)
        assert_equal(w3, expected)

        w4 = ibis.window(group_by=t.a, order_by=t.e)
        w5 = w3.combine(w4)
        expected = ibis.window(group_by=[t.g, t.a],
                               order_by=[t.f, t.e],
                               preceding=5, following=5)
        assert_equal(w5, expected)

    def test_over_auto_bind(self):
        # GH #542
        t = self.t

        w = ibis.window(group_by='g', order_by='f')

        expr = t.f.lag().over(w)

        actual_window = expr.op().args[1]
        expected = ibis.window(group_by=t.g, order_by=t.f)
        assert_equal(actual_window, expected)

    def test_window_function_bind(self):
        # GH #532
        t = self.t

        w = ibis.window(group_by=lambda x: x.g,
                        order_by=lambda x: x.f)

        expr = t.f.lag().over(w)

        actual_window = expr.op().args[1]
        expected = ibis.window(group_by=t.g, order_by=t.f)
        assert_equal(actual_window, expected)

    def test_auto_windowize_analysis_bug(self):
        # GH #544
        t = self.con.table('airlines')

        def metric(x):
            return x.arrdelay.mean().name('avg_delay')

        annual_delay = (t[t.dest.isin(['JFK', 'SFO'])]
                        .group_by(['dest', 'year'])
                        .aggregate(metric))
        what = annual_delay.group_by('dest')
        enriched = what.mutate(grand_avg=annual_delay.avg_delay.mean())

        expr = (annual_delay.avg_delay.mean().name('grand_avg')
                .over(ibis.window(group_by=annual_delay.dest)))
        expected = annual_delay[annual_delay, expr]

        assert_equal(enriched, expected)

    def test_mutate_sorts_keys(self):
        t = self.con.table('airlines')
        m = t.arrdelay.mean()
        g = t.group_by('dest')

        result = g.mutate(zzz=m, yyy=m, ddd=m, ccc=m, bbb=m, aaa=m)

        expected = g.mutate([m.name('aaa'), m.name('bbb'), m.name('ccc'),
                             m.name('ddd'), m.name('yyy'), m.name('zzz')])

        assert_equal(result, expected)

    def test_window_bind_to_table(self):
        w = ibis.window(group_by='g', order_by=ibis.desc('f'))

        w2 = w.bind(self.t)
        expected = ibis.window(group_by=self.t.g,
                               order_by=ibis.desc(self.t.f))

        assert_equal(w2, expected)

    def test_preceding_following_validate(self):
        # these all work
        [
            ibis.window(preceding=0),
            ibis.window(following=0),
            ibis.window(preceding=0, following=0),
            ibis.window(preceding=(None, 4)),
            ibis.window(preceding=(10, 4)),
            ibis.window(following=(4, None)),
            ibis.window(following=(4, 10))
        ]

        # these are ill-specified
        error_cases = [
            lambda: ibis.window(preceding=(1, 3)),
            lambda: ibis.window(preceding=(3, 1), following=2),
            lambda: ibis.window(preceding=(3, 1), following=(2, 4)),
            lambda: ibis.window(preceding=-1),
            lambda: ibis.window(following=-1),
            lambda: ibis.window(preceding=(-1, 2)),
            lambda: ibis.window(following=(2, -1))
        ]

        for i, case in enumerate(error_cases):
            with self.assertRaises(Exception):
                case()

    def test_window_equals(self):
        pass
