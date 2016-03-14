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

import os
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
            (s.cast('float'), sa.cast(sa_s, sa.types.REAL))
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
        from datetime import datetime

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

            # there could be pathological failure at midnight somewhere, but
            # that's okay
            (ibis.now().strftime('%Y%m%d %H'),
             datetime.utcnow().strftime('%Y%m%d %H'))
        ]
        self._check_e2e_cases(cases)

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

    def test_div_floordiv(self):
        cases = [
            (L(7) / L(2), 3.5),
            (L(7) // L(2), 3),
            (L(7).floordiv(2), 3),
            (L(2).rfloordiv(7), 3),
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

    def test_nullifzero(self):
        cases = [
            (L(0).nullifzero(), None),
            (L(5.5).nullifzero(), 5.5),
        ]
        self._check_e2e_cases(cases)

    def test_string_length(self):
        cases = [
            (L('foo_bar').length(), 7),
            (L('').length(), 0),
        ]
        self._check_e2e_cases(cases)

    def test_string_substring(self):
        cases = [
            (L('foo_bar').left(3), 'foo'),
            (L('foo_bar').right(3), 'bar'),

            (L('foo_bar').substr(0, 3), 'foo'),
            (L('foo_bar').substr(4, 3), 'bar'),
            (L('foo_bar').substr(1), 'oo_bar'),
        ]
        self._check_e2e_cases(cases)

    def test_string_strip(self):
        cases = [
            (L('   foo   ').lstrip(), 'foo   '),
            (L('   foo   ').rstrip(), '   foo'),
            (L('   foo   ').strip(), 'foo'),
        ]
        self._check_e2e_cases(cases)

    def test_string_upper_lower(self):
        cases = [
            (L('foo').upper(), 'FOO'),
            (L('FOO').lower(), 'foo'),
        ]
        self._check_e2e_cases(cases)

    def test_string_contains(self):
        cases = [
            (L('foobar').contains('bar'), True),
            (L('foobar').contains('foo'), True),
            (L('foobar').contains('baz'), False),
        ]
        self._check_e2e_cases(cases)

    def test_string_find(self):
        cases = [
            (L('foobar').find('bar'), 3),
            (L('foobar').find('baz'), -1),
        ]
        self._check_e2e_cases(cases)

    def test_string_like(self):
        cases = [
            (L('foobar').like('%bar'), True),
            (L('foobar').like('foo%'), True),
            (L('foobar').like('%baz%'), False),
        ]
        self._check_e2e_cases(cases)

    def test_str_replace(self):
        cases = [
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

    def test_numeric_builtins_work(self):
        t = self.alltypes
        d = t.double_col

        exprs = [
            d.fillna(0),
        ]
        self._execute_projection(t, exprs)

    def test_misc_builtins_work(self):
        t = self.alltypes
        d = t.double_col

        exprs = [
            (d > 20).ifelse(10, -20),
            (d > 20).ifelse(10, -20).abs(),

            # tier and histogram
            d.bucket([0, 10, 25, 50, 100]),
            d.bucket([0, 10, 25, 50], include_over=True),
            d.bucket([0, 10, 25, 50], include_over=True, close_extreme=False),
            d.bucket([10, 25, 50, 100], include_under=True),
        ]
        self._execute_projection(t, exprs)

    def test_category_label(self):
        t = self.alltypes
        d = t.double_col

        bucket = d.bucket([0, 10, 25, 50, 100])

        exprs = [
            bucket.label(['a', 'b', 'c', 'd'])
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
            table.bool_col.any(),
            table.bool_col.all(),
            table.bool_col.notany(),
            table.bool_col.notall(),

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
        self._execute_aggregation(table, exprs)

    def test_distinct_aggregates(self):
        table = self.alltypes.limit(100)

        exprs = [
            table.double_col.nunique()
        ]
        self._execute_aggregation(table, exprs)

    def test_not_exists_works(self):
        t = self.alltypes
        t2 = t.view()

        expr = t[-(t.string_col == t2.string_col).any()]
        expr.execute()

    def test_interactive_repr_shows_error(self):
        # #591. Doing this in SQLite because so many built-in functions are not
        # available
        import ibis.config as config

        expr = self.alltypes.double_col.approx_nunique()

        with config.option_context('interactive', True):
            result = repr(expr)
            assert 'no translator rule' in result.lower()

    def test_subquery_invokes_sqlite_compiler(self):
        t = self.alltypes

        expr = (t.mutate(d=t.double_col.fillna(0))
                .limit(1000)
                .group_by('string_col')
                .size())
        expr.execute()

    def _execute_aggregation(self, table, exprs):
        agg_exprs = [expr.name('e%d' % i)
                     for i, expr in enumerate(exprs)]

        agged_table = table.aggregate(agg_exprs)
        agged_table.execute()

    def _execute_projection(self, table, exprs):
        agg_exprs = [expr.name('e%d' % i)
                     for i, expr in enumerate(exprs)]

        proj = table.projection(agg_exprs)
        proj.execute()

    def test_filter_has_sqla_table(self):
        t = self.alltypes
        pred = t.year == 2010
        filt = t.filter(pred).sort_by('float_col').float_col
        s = filt.execute()
        result = s.squeeze().reset_index(drop=True)
        expected = t.execute().query(
            'year == 2010'
        ).sort('float_col').float_col

        assert len(result) == len(expected)

    def test_column_access_after_sort(self):
        t = self.alltypes
        expr = t.sort_by('float_col').string_col

        # it works!
        expr.execute(limit=10)

    def test_materialized_join(self):
        path = '__ibis_tmp_{0}.db'.format(ibis.util.guid())

        con = ibis.sqlite.connect(path, create=True)

        try:
            con.raw_sql("create table mj1 (id1 integer, val1 real)")
            con.raw_sql("insert into mj1 values (1, 10), (2, 20)")
            con.raw_sql("create table mj2 (id2 integer, val2 real)")
            con.raw_sql("insert into mj2 values (1, 15), (2, 25)")

            t1 = con.table('mj1')
            t2 = con.table('mj2')
            joined = t1.join(t2, t1.id1 == t2.id2).materialize()
            result = joined.val2.execute()
            assert len(result) == 2
        finally:
            os.remove(path)
