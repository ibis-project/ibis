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


from ibis import window
import ibis

from ibis.impala.compiler import to_sql
from ibis.expr.tests.mocks import BasicTestCase
from ibis.compat import unittest
from ibis.tests.util import assert_equal
import ibis.common as com


class TestWindowFunctions(BasicTestCase, unittest.TestCase):

    def _check_sql(self, expr, expected):
        result = to_sql(expr)
        assert result == expected

    def test_aggregate_in_projection(self):
        t = self.con.table('alltypes')
        proj = t[t, (t.f / t.f.sum()).name('normed_f')]

        expected = """\
SELECT *, `f` / sum(`f`) OVER () AS `normed_f`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_add_default_order_by(self):
        t = self.con.table('alltypes')

        first = t.f.first().name('first')
        last = t.f.last().name('last')
        lag = t.f.lag().name('lag')
        diff = (t.f.lead() - t.f).name('fwd_diff')
        lag2 = t.f.lag().over(window(order_by=t.d)).name('lag2')
        grouped = t.group_by('g')
        proj = grouped.mutate([lag, diff, first, last, lag2])
        expected = """\
SELECT *, lag(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `lag`,
       lead(`f`) OVER (PARTITION BY `g` ORDER BY `f`) - `f` AS `fwd_diff`,
       first_value(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `first`,
       last_value(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `last`,
       lag(`f`) OVER (PARTITION BY `g` ORDER BY `d`) AS `lag2`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_window_frame_specs(self):
        t = self.con.table('alltypes')

        ex_template = """\
SELECT sum(`d`) OVER (ORDER BY `f` {0}) AS `foo`
FROM alltypes"""

        cases = [
            (window(preceding=0),
             'range between current row and unbounded following'),

            (window(following=0),
             'range between unbounded preceding and current row'),

            (window(preceding=5),
             'rows between 5 preceding and unbounded following'),
            (window(preceding=5, following=0),
             'rows between 5 preceding and current row'),
            (window(preceding=5, following=2),
             'rows between 5 preceding and 2 following'),
            (window(following=2),
             'rows between unbounded preceding and 2 following'),
            (window(following=2, preceding=0),
             'rows between current row and 2 following'),
            (window(preceding=5),
             'rows between 5 preceding and unbounded following'),
            (window(following=[5, 10]),
             'rows between 5 following and 10 following'),
            (window(preceding=[10, 5]),
             'rows between 10 preceding and 5 preceding'),

            # # cumulative windows
            (ibis.cumulative_window(),
             'range between unbounded preceding and current row'),

            # # trailing windows
            (ibis.trailing_window(10),
             'rows between 10 preceding and current row'),
        ]

        for w, frame in cases:
            w2 = w.order_by(t.f)
            expr = t.projection([t.d.sum().over(w2).name('foo')])
            expected = ex_template.format(frame.upper())
            self._check_sql(expr, expected)

    def test_cumulative_functions(self):
        t = self.con.table('alltypes')

        w = ibis.window(order_by=t.d)
        exprs = [
            (t.f.cumsum().over(w), t.f.sum().over(w)),
            (t.f.cummin().over(w), t.f.min().over(w)),
            (t.f.cummax().over(w), t.f.max().over(w)),
            (t.f.cummean().over(w), t.f.mean().over(w)),
        ]

        for cumulative, static in exprs:
            actual = cumulative.name('foo')
            expected = static.over(ibis.cumulative_window()).name('foo')

            expr1 = t.projection(actual)
            expr2 = t.projection(expected)

            self._compare_sql(expr1, expr2)

    def _compare_sql(self, e1, e2):
        s1 = to_sql(e1)
        s2 = to_sql(e2)
        assert s1 == s2

    def test_nested_analytic_function(self):
        t = self.con.table('alltypes')

        w = window(order_by=t.f)
        expr = (t.f - t.f.lag()).lag().over(w).name('foo')
        result = t.projection([expr])
        expected = """\
SELECT lag(`f` - lag(`f`) OVER (ORDER BY `f`)) \
OVER (ORDER BY `f`) AS `foo`
FROM alltypes"""
        self._check_sql(result, expected)

    def test_rank_functions(self):
        t = self.con.table('alltypes')

        proj = t[t.g, t.f.rank().name('minr'),
                 t.f.dense_rank().name('denser')]
        expected = """\
SELECT `g`, rank() OVER (ORDER BY `f`) - 1 AS `minr`,
       dense_rank() OVER (ORDER BY `f`) - 1 AS `denser`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_multiple_windows(self):
        t = self.con.table('alltypes')

        w = window(group_by=t.g)

        expr = t.f.sum().over(w) - t.f.sum()
        proj = t.projection([t.g, expr.name('result')])

        expected = """\
SELECT `g`, sum(`f`) OVER (PARTITION BY `g`) - sum(`f`) OVER () AS `result`
FROM alltypes"""
        self._check_sql(proj, expected)

    def test_order_by_desc(self):
        t = self.con.table('alltypes')

        w = window(order_by=ibis.desc(t.f))

        proj = t[t.f, ibis.row_number().over(w).name('revrank')]
        expected = """\
SELECT `f`, row_number() OVER (ORDER BY `f` DESC) - 1 AS `revrank`
FROM alltypes"""
        self._check_sql(proj, expected)

        expr = (t.group_by('g')
                .order_by(ibis.desc(t.f))
                [t.d.lag().name('foo'), t.a.max()])
        expected = """\
SELECT lag(`d`) OVER (PARTITION BY `g` ORDER BY `f` DESC) AS `foo`,
       max(`a`) OVER (PARTITION BY `g` ORDER BY `f` DESC) AS `max`
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
SELECT *, row_number() OVER (PARTITION BY `g` ORDER BY `f`) - 1 AS `foo`
FROM alltypes"""
        self._check_sql(expr, expected)

    def test_unsupported_aggregate_functions(self):
        t = self.con.table('alltypes')
        w = ibis.window(order_by=t.d)

        exprs = [
            t.f.approx_nunique(),
            t.f.approx_median(),
            t.g.group_concat(),
        ]

        for expr in exprs:
            with self.assertRaises(com.TranslationError):
                proj = t.projection([expr.over(w).name('foo')])
                to_sql(proj)

    def test_propagate_nested_windows(self):
        # GH #469
        t = self.con.table('alltypes')

        w = ibis.window(group_by=t.g, order_by=t.f)

        col = (t.f - t.f.lag()).lag()

        # propagate down here!
        result = col.over(w)
        ex_expr = (t.f - t.f.lag().over(w)).lag().over(w)
        assert_equal(result, ex_expr)

        expr = t.projection(col.over(w).name('foo'))
        expected = """\
SELECT lag(`f` - lag(`f`) OVER (PARTITION BY `g` ORDER BY `f`)) \
OVER (PARTITION BY `g` ORDER BY `f`) AS `foo`
FROM alltypes"""
        self._check_sql(expr, expected)

    def test_math_on_windowed_expr(self):
        # Window clause may not be found at top level of expression
        pass

    def test_group_by_then_different_sort_orders(self):
        pass
