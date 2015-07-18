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

from ibis.sql.compiler import to_sql
from ibis.expr.tests.mocks import BasicTestCase
from ibis.compat import unittest
import ibis.common as com


class TestWindowFunctions(BasicTestCase, unittest.TestCase):

    def _check_sql(self, expr, expected):
        result = to_sql(expr)
        assert result == expected

    def test_aggregate_in_projection(self):
        t = self.con.table('alltypes')
        proj = t[t, (t.f / t.f.sum()).name('normed_f')]

        expected = """\
SELECT *, f / sum(f) OVER () AS `normed_f`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_add_default_order_by(self):
        t = self.con.table('alltypes')

        first = t.f.first().name('first')
        last = t.f.last().name('last')
        lag = t.f.lag().name('lag')
        diff = (t.f.lead() - t.f).name('fwd_diff')
        lag2 = t.f.lag().over(ibis.window(order_by=t.d)).name('lag2')
        grouped = t.group_by('g')
        proj = grouped.mutate([lag, diff, first, last, lag2])
        expected = """\
SELECT *, lag(f) OVER (PARTITION BY g ORDER BY f) AS `lag`,
       lead(f) OVER (PARTITION BY g ORDER BY f) - f AS `fwd_diff`,
       first_value(f) OVER (PARTITION BY g ORDER BY f) AS `first`,
       last_value(f) OVER (PARTITION BY g ORDER BY f) AS `last`,
       lag(f) OVER (PARTITION BY g ORDER BY d) AS `lag2`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_nested_analytic_function(self):
        t = self.con.table('alltypes')

        w = ibis.window(order_by=t.f)
        expr = (t.f - t.f.lag()).lag().over(w).name('foo')
        result = t.projection([expr])
        expected = """\
SELECT lag(f - lag(f) OVER (ORDER BY f)) \
OVER (ORDER BY f) AS `foo`
FROM alltypes"""
        self._check_sql(result, expected)

    def test_rank_functions(self):
        t = self.con.table('alltypes')

        proj = t[t.g, t.f.rank().name('minr'),
                 t.f.dense_rank().name('denser')]
        expected = """\
SELECT g, rank() OVER (ORDER BY f) - 1 AS `minr`,
       dense_rank() OVER (ORDER BY f) - 1 AS `denser`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_multiple_windows(self):
        t = self.con.table('alltypes')

        w = ibis.window(group_by=t.g)

        expr = t.f.sum().over(w) - t.f.sum()
        proj = t.projection([t.g, expr.name('result')])

        expected = """\
SELECT g, sum(f) OVER (PARTITION BY g) - sum(f) OVER () AS `result`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_order_by_desc(self):
        t = self.con.table('alltypes')

        w = ibis.window(order_by=ibis.desc(t.f))

        proj = t[t.f, ibis.row_number().over(w).name('revrank')]
        expected = """\
SELECT f, row_number() OVER (ORDER BY f DESC) - 1 AS `revrank`
FROM alltypes"""
        self._check_sql(proj, expected)

        expr = (t.group_by('g')
                .order_by(ibis.desc(t.f))
                [t.d.lag().name('foo'), t.a.max()])
        expected = """\
SELECT lag(d) OVER (PARTITION BY g ORDER BY f DESC) AS `foo`,
       max(a) OVER (PARTITION BY g ORDER BY f DESC) AS `max`
FROM alltypes"""
        self._check_sql(expr, expected)

    def test_row_number_requires_order_by(self):
        t = self.con.table('alltypes')

        with self.assertRaises(com.ExpressionError):
            (t.group_by(t.g)
             .mutate(ibis.row_number().name('foo')))

        expr = (t.group_by(t.g)
                .order_by(t.f)
                .mutate(ibis.row_number().name('foo')))

        expected = """\
SELECT *, row_number() OVER (PARTITION BY g ORDER BY f) - 1 AS `foo`
FROM alltypes"""
        self._check_sql(expr, expected)

    def test_math_on_windowed_expr(self):
        # Window clause may not be found at top level of expression
        pass

    def test_group_by_then_different_sort_orders(self):
        pass

    def test_cumulative_window(self):
        pass

    def test_trailing_window(self):
        pass

    def test_cumulative_functions(self):
        pass

    def test_off_center_window(self):
        w1 = ibis.window(following=(5, 10))
        w2 = ibis.window(preceding=(10, 5))
