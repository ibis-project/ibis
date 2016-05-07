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

import math

import pytest  # noqa

from .common import PostgreSQLTests
from ibis.compat import unittest
from ibis import literal as L
import ibis.expr.types as ir
import ibis

import sqlalchemy as sa

import pandas as pd
import pandas.util.testing as tm


class TestPostgreSQLFunctions(PostgreSQLTests, unittest.TestCase):

    def test_cast(self):
        at = self._to_sqla(self.alltypes)

        d = self.alltypes.double_col
        s = self.alltypes.string_col

        sa_d = at.c.double_col
        sa_s = at.c.string_col

        cases = [
            (d.cast('int8'), sa.cast(sa_d, sa.SMALLINT)),
            (d.cast('int16'), sa.cast(sa_d, sa.SMALLINT)),
            (s.cast('double'), sa.cast(sa_s, sa.FLOAT)),
            (s.cast('float'), sa.cast(sa_s, sa.REAL))
        ]
        self._check_expr_cases(cases)

    @pytest.mark.xfail(raises=AssertionError, reason='NYI')
    def test_decimal_cast(self):
        assert False

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

        cases = [
            (tc_casted, at.c.timestamp_col),
            (ic_casted,
             sa.func.timezone('UTC', sa.func.to_timestamp(at.c.int_col)))
        ]
        self._check_expr_cases(cases)

    def test_timestamp_functions(self):
        from datetime import datetime

        v = L('2015-09-01 14:48:05.359').cast('timestamp')
        vt = datetime(
            year=2015, month=9, day=1,
            hour=14, minute=48, second=5, microsecond=359000
        )

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
            (v.strftime('%Y%m%d %H'), vt.strftime('%Y%m%d %H')),

            # test quoting behavior
            (v.strftime('DD BAR %w FOO "DD"'),
             vt.strftime('DD BAR %w FOO "DD"')),
            (v.strftime('DD BAR %w FOO "D'),
             vt.strftime('DD BAR %w FOO "D')),
            (v.strftime('DD BAR "%w" FOO "D'),
             vt.strftime('DD BAR "%w" FOO "D')),
            (v.strftime('DD BAR "%d" FOO "D'),
             vt.strftime('DD BAR "%d" FOO "D')),
            (v.strftime('DD BAR "%c" FOO "D'),
             vt.strftime('DD BAR "%c" FOO "D')),
            (v.strftime('DD BAR "%x" FOO "D'),
             vt.strftime('DD BAR "%x" FOO "D')),
            (v.strftime('DD BAR "%X" FOO "D'),
             vt.strftime('DD BAR "%X" FOO "D'))
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

            # TODO: this should really be double
            (L(1.2345).typeof(), 'numeric'),
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

    def test_string_pad(self):
        cases = [
            (L('foo').lpad(6, ' '), '   foo'),
            (L('foo').rpad(6, ' '), 'foo   '),
        ]
        self._check_e2e_cases(cases)

    def test_string_reverse(self):
        cases = [
            (L('foo').reverse(), 'oof'),
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
            (L('100%').contains('%'), True),
            (L('a_b_c').contains('_'), True)
        ]
        self._check_e2e_cases(cases)

    def test_capitalize(self):
        cases = [
            (L('foo bar foo').capitalize(), 'Foo Bar Foo'),
            (L('foobar Foo').capitalize(), 'Foobar Foo'),
        ]
        self._check_e2e_cases(cases)

    def test_repeat(self):
        cases = [
            (L('bar ').repeat(3), 'bar bar bar '),
        ]
        self._check_e2e_cases(cases)

    def test_re_replace(self):
        cases = [
            (
                L('fudge|||chocolate||candy').re_replace('\\|{2,3}', ', '),
                'fudge, chocolate, candy'
            )
        ]
        self._check_e2e_cases(cases)

    def test_translate(self):
        cases = [
            (L('faab').translate('a', 'b'), 'fbbb'),
        ]
        self._check_e2e_cases(cases)

    def test_find_in_set(self):
        cases = [
            (L('a').find_in_set(list('abc')), 0),
            (L('b').find_in_set(list('abc')), 1),
        ]
        self._check_e2e_cases(cases)

    def test_find_in_set_null_scalar(self):
        cases = [
            (L(None).cast('string').find_in_set(['a', 'b', None]), 2),
        ]
        self._check_e2e_cases(cases)

    def test_isnull_notnull(self):
        cases = [
            (L(None).isnull(), True),
            (L(1).isnull(), False),
            (L(None).notnull(), False),
            (L(1).notnull(), True),
        ]
        self._check_e2e_cases(cases)

    def test_string_functions(self):
        cases = [
            (L('foobar').find('bar'), 3),
            (L('foobar').find('baz'), -1),

            (L('foobar').like('%bar'), True),
            (L('foobar').like('foo%'), True),
            (L('foobar').like('%baz%'), False),

            (L('foobarfoo').replace('foo', 'H'), 'HbarH'),
            (L('a').ascii_str(), ord('a'))
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
            (L(5.556).ceil(), 6.0),
            (L(5.556).floor(), 5.0),
            (L(5.556).exp(), math.exp(5.556)),
            (L(5.556).sign(), 1),
            (L(-5.556).sign(), -1),
            (L(0).sign(), 0),
            (L(5.556).sqrt(), math.sqrt(5.556)),
            (L(5.556).log(2), math.log(5.556, 2)),
            (L(5.556).ln(), math.log(5.556)),
            (L(5.556).log2(), math.log(5.556, 2)),
            (L(5.556).log10(), math.log10(5.556)),
        ]
        self._check_e2e_cases(cases)

    def test_regexp(self):
        v = L('abcd')
        v2 = L('1222')
        cases = [
            (v.re_search('[a-z]'), True),
            (v.re_search('[\d]+'), False),
            (v2.re_search('[\d]+'), True),
        ]
        self._check_e2e_cases(cases)

    def test_regexp_extract(self):
        cases = [
            (L('abcd').re_extract('([a-z]+)', 0), 'abcd'),
            (L('abcd').re_extract('(ab)(cd)', 1), 'cd'),

            # valid group number but no match => empty string
            (L('abcd').re_extract('(\d)', 0), ''),

            # match but not a valid group number => NULL
            (L('abcd').re_extract('abcd', 3), None),
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
        cases = [
            (ibis.coalesce(5, None, 4), 5),
            (ibis.coalesce(ibis.NA, 4, ibis.NA), 4),
            (ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14),
        ]
        self._check_e2e_cases(cases)

    @pytest.mark.xfail(raises=TypeError, reason='Ambiguous argument types')
    def test_coalesce_all_na(self):
        NA = ibis.NA
        int8_na = ibis.NA.cast('int8')
        cases = [
            (ibis.coalesce(NA, NA), NA),
            (ibis.coalesce(NA, NA, NA.cast('double')), NA),
            (ibis.coalesce(int8_na, int8_na, int8_na), int8_na),
        ]
        self._check_e2e_cases(cases)


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

    @pytest.mark.xfail(
        raises=sa.exc.ProgrammingError,
        reason='union not working yet'
    )
    def test_union(self):
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
            d.var(),
            d.std(),
            d.var(how='sample'),
            d.std(how='pop'),

            table.bool_col.count(where=cond),
            d.sum(where=cond),
            d.mean(where=cond),
            d.min(where=cond),
            d.max(where=cond),
            d.var(where=cond),
            d.std(where=cond),
            d.var(where=cond, how='sample'),
            d.std(where=cond, how='pop'),

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
        # #591. Doing this in PostgreSQL because so many built-in functions are
        # not available
        import ibis.config as config

        expr = self.alltypes.double_col.approx_nunique()

        with config.option_context('interactive', True):
            result = repr(expr)
            assert 'no translator rule' in result.lower()

    def test_subquery_invokes_postgresql_compiler(self):
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

    def test_simple_window(self):
        t = self.alltypes
        df = t.execute()
        for func in 'mean sum min max'.split():
            f = getattr(t.double_col, func)
            df_f = getattr(df.double_col, func)
            result = t.projection([(t.double_col - f()).name('double_col')]).execute().double_col
            expected = df.double_col - df_f()
            tm.assert_series_equal(result, expected)

    def test_rolling_window(self):
        t = self.alltypes
        df = t[['double_col', 'timestamp_col']].execute().sort_values('timestamp_col').reset_index(drop=True)
        window = ibis.window(
            order_by=t.timestamp_col,
            preceding=6,
            following=0
        )
        for func in 'mean sum min max'.split():
            f = getattr(t.double_col, func)
            df_f = getattr(df.double_col.rolling(7, min_periods=0), func)
            result = t.projection([f().over(window).name('double_col')]).execute().double_col
            expected = df_f()
            tm.assert_series_equal(result, expected)

    def test_partitioned_window(self):
        t = self.alltypes
        df = t.execute()
        window = ibis.window(
            group_by=t.string_col,
            order_by=t.timestamp_col,
            preceding=6,
            following=0,
        )

        def roller(func):
            def rolled(df):
                torder = df.sort_values('timestamp_col')
                rolling = torder.double_col.rolling(7, min_periods=0)
                return getattr(rolling, func)()
            return rolled

        for func in 'mean sum min max'.split():
            f = getattr(t.double_col, func)
            expr = f().over(window).name('double_col')
            result = t.projection([expr]).execute().double_col
            expected = df.groupby('string_col').apply(
                roller(func)
            ).reset_index(drop=True)
            tm.assert_series_equal(result, expected)

    def test_cumulative_simple_window(self):
        t = self.alltypes
        df = t.execute()
        for func in 'sum min max'.split():
            f = getattr(t.double_col, func)
            expr = t.projection([(t.double_col - f().over(ibis.cumulative_window())).name('double_col')])
            result = expr.execute().double_col
            expected = df.double_col - getattr(df.double_col, 'cum%s' % func)()
            tm.assert_series_equal(result, expected)

    def test_cumulative_partitioned_window(self):
        t = self.alltypes
        df = t.execute().sort_values('string_col').reset_index(drop=True)
        window = ibis.cumulative_window(group_by=t.string_col)
        for func in 'sum min max'.split():
            f = getattr(t.double_col, func)
            expr = t.projection([(t.double_col - f().over(window)).name('double_col')])
            result = expr.execute().double_col
            expected = df.groupby(df.string_col).double_col.transform(lambda c: c - getattr(c, 'cum%s' % func)())
            tm.assert_series_equal(result, expected)

    def test_cumulative_ordered_window(self):
        t = self.alltypes
        df = t.execute().sort_values('timestamp_col').reset_index(drop=True)
        window = ibis.cumulative_window(order_by=t.timestamp_col)
        for func in 'sum min max'.split():
            f = getattr(t.double_col, func)
            expr = t.projection([(t.double_col - f().over(window)).name('double_col')])
            result = expr.execute().double_col
            expected = df.double_col - getattr(df.double_col, 'cum%s' % func)()
            tm.assert_series_equal(result, expected)

    def test_cumulative_partitioned_ordered_window(self):
        t = self.alltypes
        df = t.execute().sort_values(['string_col', 'timestamp_col']).reset_index(drop=True)
        window = ibis.cumulative_window(order_by=t.timestamp_col, group_by=t.string_col)
        for func in 'sum min max'.split():
            f = getattr(t.double_col, func)
            expr = t.projection([(t.double_col - f().over(window)).name('double_col')])
            result = expr.execute().double_col
            expected = df.groupby(df.string_col).double_col.transform(lambda c: c - getattr(c, 'cum%s' % func)())
            tm.assert_series_equal(result, expected)
