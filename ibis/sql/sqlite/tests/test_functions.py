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

import pytest  # noqa

from .common import SQLiteTests
from ibis.compat import unittest
from ibis import literal as L
import ibis.expr.types as ir
import ibis

import sqlalchemy as sa


class TestSQLiteFunctions(SQLiteTests, unittest.TestCase):

    def test_cast(self):
        at = self._to_sqla(self.alltypes)

        d = self.alltypes.double_col
        s = self.alltypes.string_col

        sa_d = at.c.double_col
        sa_s = at.c.string_col

        cases = [
            (d.cast('int8'), sa.cast(sa_d, sa.types.SMALLINT)),
            (s.cast('double'), sa.cast(sa_s, sa.types.REAL)),
        ]
        self._check_expr_cases(cases)

    def test_decimal_cast(self):
        pass

    def test_timestamp_cast_noop(self):
        # See GH #592

        at = self._to_sqla(self.alltypes)

        tc = self.alltypes.timestamp_col
        ic = self.alltypes.int_col

        tc_casted = tc.cast('timestamp')
        ic_casted = ic.cast('timestamp')

        # Logically, it's a timestamp
        assert isinstance(tc_casted, ir.TimestampArray)
        assert isinstance(ic_casted, ir.TimestampArray)

        # But it's a no-op when translated to SQLAlchemy
        cases = [
            (tc_casted, at.c.timestamp_col),
            (ic_casted, at.c.int_col)
        ]
        self._check_expr_cases(cases)

    def test_timestamp_functions(self):
        v = L('2015-09-01 14:48:05.359').cast('timestamp')

        cases = [
            (v.strftime('%Y%m%d'), '20150901'),

            (v.year(), 2015),
            (v.month(), 9),
            (v.day(), 1),
            (v.hour(), 14),
            (v.minute(), 48),
            (v.second(), 5),
            (v.millisecond(), 359),
        ]
        self._check_e2e_cases(cases)

    def test_timestamp_now(self):
        pass

    def test_binary_arithmetic(self):
        cases = [
            (L(3) + L(4), 7),
            (L(3) - L(4), -1),
            (L(3) * L(4), 12),
            (L(12) / L(4), 3),
            # (L(12) ** L(2), 144),
            (L(12) % L(5), 2)
        ]
        self._check_e2e_cases(cases)

    def test_typeof(self):
        cases = [
            (L('foo_bar').typeof(), 'text'),
            (L(5).typeof(), 'integer'),
            (ibis.NA.typeof(), 'null'),
            (L(1.2345).typeof(), 'real'),
        ]
        self._check_e2e_cases(cases)

    def test_string_functions(self):
        cases = [
            (L('foo_bar').length(), 7),

            (L('foo_bar').left(3), 'foo'),
            (L('foo_bar').right(3), 'bar'),

            (L('foo_bar').substr(0, 3), 'foo'),
            (L('foo_bar').substr(4, 3), 'bar'),
            (L('foo_bar').substr(1), 'oo_bar'),

            (L('   foo   ').lstrip(), 'foo   '),
            (L('   foo   ').rstrip(), '   foo'),
            (L('   foo   ').strip(), 'foo'),

            (L('foo').upper(), 'FOO'),
            (L('FOO').lower(), 'foo'),

            (L('foobar').find('bar'), 3),
            (L('foobar').find('baz'), -1),

            (L('foobar').like('%bar'), True),
            (L('foobar').like('foo%'), True),
            (L('foobar').like('%baz%'), False),

            (L('foobar').contains('bar'), True),
            (L('foobar').contains('foo'), True),
            (L('foobar').contains('baz'), False),

            (L('foobarfoo').replace('foo', 'H'), 'HbarH'),
        ]
        self._check_e2e_cases(cases)

    def test_math_functions(self):
        cases = [
            (L(-5).abs(), 5),
            (L(5).abs(), 5),
            (ibis.least(L(5), L(10), L(1)), 1),
            (ibis.greatest(L(5), L(10), L(1)), 10),

            (L(5.5).round(), 6.0),
            (L(5.556).round(2), 5.56),
        ]
        self._check_e2e_cases(cases)

    def test_regexp(self):
        pytest.skip('NYI: Requires adding regex udf with sqlite3')

        v = L('abcd')
        v2 = L('1222')
        cases = [
            (v.re_search('[a-z]'), True),
            (v.re_search('[\d]+'), False),
            (v2.re_search('[\d]+'), True),
        ]
        self._check_e2e_cases(cases)

    def test_fillna_nullif(self):
        cases = [
            (ibis.NA.fillna(5), 5),
            (L(5).fillna(10), 5),
            (L(5).nullif(5), None),
            (L(10).nullif(5), 10),
        ]
        self._check_e2e_cases(cases)

    def test_coalesce(self):
        pass

    def test_misc_builtins_work(self):
        t = self.alltypes

        d = t.double_col

        exprs = [
            (d > 20).ifelse(10, -20),
            (d > 20).ifelse(10, -20).abs(),
        ]
        self._execute_projection(t, exprs)

    def test_union(self):
        pytest.skip('union not working yet')

        t = self.alltypes

        expr = (t.group_by('string_col')
                .aggregate(t.double_col.sum().name('foo'))
                .sort_by('string_col'))

        t1 = expr.limit(4)
        t2 = expr.limit(4, offset=4)
        t3 = expr.limit(8)

        result = t1.union(t2).execute()
        expected = t3.execute()

        assert (result.string_col == expected.string_col).all()

    def test_aggregations_execute(self):
        table = self.alltypes.limit(100)

        d = table.double_col
        s = table.string_col

        cond = table.string_col.isin(['1', '7'])

        exprs = [
            table.bool_col.count(),

            d.sum(),
            d.mean(),
            d.min(),
            d.max(),

            table.bool_col.count(where=cond),
            d.sum(where=cond),
            d.mean(where=cond),
            d.min(where=cond),
            d.max(where=cond),

            s.group_concat(),
        ]

        agg_exprs = [expr.name('e%d' % i)
                     for i, expr in enumerate(exprs)]

        agged_table = table.aggregate(agg_exprs)
        agged_table.execute()

    def _execute_projection(self, table, exprs):
        agg_exprs = [expr.name('e%d' % i)
                     for i, expr in enumerate(exprs)]

        proj = table.projection(agg_exprs)
        proj.execute()
