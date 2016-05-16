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

import pytest

import pandas as pd

import ibis

from ibis import literal as L
from ibis.compat import unittest, StringIO, Decimal
from ibis.expr.datatypes import Category
from ibis.expr.tests.mocks import MockConnection
from ibis.impala.compiler import ImpalaExprTranslator, to_sql, ImpalaContext
from ibis.sql.tests.test_compiler import ExprTestCases
from ibis.impala.tests.common import ImpalaE2E
import ibis.expr.types as ir
import ibis.expr.api as api


def approx_equal(a, b, eps):
    assert abs(a - b) < eps


class ExprSQLTest(object):

    def _check_expr_cases(self, cases, context=None, named=False):
        for expr, expected in cases:
            repr(expr)
            result = self._translate(expr, named=named, context=context)
            assert result == expected

    def _translate(self, expr, named=False, context=None):
        translator = ImpalaExprTranslator(expr, context=context, named=named)
        return translator.get_result()


class TestValueExprs(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

        self.int_cols = ['a', 'b', 'c', 'd']
        self.bool_cols = ['h']
        self.float_cols = ['e', 'f']

    def _check_literals(self, cases):
        for value, expected in cases:
            lit_expr = L(value)
            result = self._translate(lit_expr)
            assert result == expected

    def test_string_literals(self):
        cases = [
            ('simple', "'simple'"),
            ('I can\'t', "'I can\\'t'"),
            ('An "escape"', "'An \"escape\"'")
        ]

        for value, expected in cases:
            lit_expr = L(value)
            result = self._translate(lit_expr)
            assert result == expected

    def test_decimal_builtins(self):
        t = self.con.table('tpch_lineitem')
        col = t.l_extendedprice
        cases = [
            (col.precision(), 'precision(`l_extendedprice`)'),
            (col.scale(), 'scale(`l_extendedprice`)'),
        ]
        self._check_expr_cases(cases)

    def test_number_boolean_literals(self):
        cases = [
            (5, '5'),
            (1.5, '1.5'),
            (True, 'TRUE'),
            (False, 'FALSE')
        ]
        self._check_literals(cases)

    def test_column_ref_table_aliases(self):
        context = ImpalaContext()

        table1 = ibis.table([
            ('key1', 'string'),
            ('value1', 'double')
        ])

        table2 = ibis.table([
            ('key2', 'string'),
            ('value and2', 'double')
        ])

        context.set_ref(table1, 't0')
        context.set_ref(table2, 't1')

        expr = table1['value1'] - table2['value and2']

        result = self._translate(expr, context=context)
        expected = 't0.`value1` - t1.`value and2`'
        assert result == expected

    def test_column_ref_quoting(self):
        schema = [('has a space', 'double')]
        table = ibis.table(schema)
        self._translate(table['has a space'], '`has a space`')

    def test_identifier_quoting(self):
        schema = [('date', 'double'), ('table', 'string')]
        table = ibis.table(schema)
        self._translate(table['date'], '`date`')
        self._translate(table['table'], '`table`')

    def test_named_expressions(self):
        a, b, g = self.table.get_columns(['a', 'b', 'g'])

        cases = [
            (g.cast('double').name('g_dub'), 'CAST(`g` AS double) AS `g_dub`'),
            (g.name('has a space'), '`g` AS `has a space`'),
            (((a - b) * a).name('expr'), '(`a` - `b`) * `a` AS `expr`')
        ]

        return self._check_expr_cases(cases, named=True)

    def test_binary_infix_operators(self):
        # For each function, verify that the generated code is what we expect
        a, b, h = self.table.get_columns(['a', 'b', 'h'])
        bool_col = a > 0

        cases = [
            (a + b, '`a` + `b`'),
            (a - b, '`a` - `b`'),
            (a * b, '`a` * `b`'),
            (a / b, '`a` / `b`'),
            (a ** b, 'pow(`a`, `b`)'),
            (a < b, '`a` < `b`'),
            (a <= b, '`a` <= `b`'),
            (a > b, '`a` > `b`'),
            (a >= b, '`a` >= `b`'),
            (a == b, '`a` = `b`'),
            (a != b, '`a` != `b`'),
            (h & bool_col, '`h` AND (`a` > 0)'),
            (h | bool_col, '`h` OR (`a` > 0)'),
            # xor is brute force
            (h ^ bool_col, '(`h` OR (`a` > 0)) AND NOT (`h` AND (`a` > 0))')
        ]
        self._check_expr_cases(cases)

    def test_binary_infix_parenthesization(self):
        a, b, c = self.table.get_columns(['a', 'b', 'c'])

        cases = [
            ((a + b) + c, '(`a` + `b`) + `c`'),
            (a.log() + c, 'ln(`a`) + `c`'),
            (b + (-(a + c)), '`b` + (-(`a` + `c`))')
        ]

        self._check_expr_cases(cases)

    def test_between(self):
        cases = [
            (self.table.f.between(0, 1), '`f` BETWEEN 0 AND 1')
        ]
        self._check_expr_cases(cases)

    def test_isnull_notnull(self):
        cases = [
            (self.table['g'].isnull(), '`g` IS NULL'),
            (self.table['a'].notnull(), '`a` IS NOT NULL'),
            ((self.table['a'] + self.table['b']).isnull(),
             '`a` + `b` IS NULL')
        ]
        self._check_expr_cases(cases)

    def test_casts(self):
        a, d, g = self.table.get_columns(['a', 'd', 'g'])
        cases = [
            (a.cast('int16'), 'CAST(`a` AS smallint)'),
            (a.cast('int32'), 'CAST(`a` AS int)'),
            (a.cast('int64'), 'CAST(`a` AS bigint)'),
            (a.cast('float'), 'CAST(`a` AS float)'),
            (a.cast('double'), 'CAST(`a` AS double)'),
            (a.cast('string'), 'CAST(`a` AS string)'),
            (d.cast('int8'), 'CAST(`d` AS tinyint)'),
            (g.cast('double'), 'CAST(`g` AS double)'),
            (g.cast('timestamp'), 'CAST(`g` AS timestamp)')
        ]
        self._check_expr_cases(cases)

    def test_misc_conditionals(self):
        a = self.table.a
        cases = [
            (a.nullif(0), 'nullif(`a`, 0)')
        ]
        self._check_expr_cases(cases)

    def test_decimal_casts(self):
        cases = [
            (L('9.9999999').cast('decimal(38,5)'),
             "CAST('9.9999999' AS decimal(38,5))"),
            (self.table.f.cast('decimal(12,2)'), "CAST(`f` AS decimal(12,2))")
        ]
        self._check_expr_cases(cases)

    def test_negate(self):
        cases = [
            (-self.table['a'], '-`a`'),
            (-self.table['f'], '-`f`'),
            (-self.table['h'], 'NOT `h`')
        ]
        self._check_expr_cases(cases)

    def test_timestamp_extract_field(self):
        fields = ['year', 'month', 'day', 'hour', 'minute',
                  'second', 'millisecond']

        cases = [(getattr(self.table.i, field)(),
                  "extract(`i`, '{0}')".format(field))
                 for field in fields]
        self._check_expr_cases(cases)

        # integration with SQL translation
        expr = self.table[self.table.i.year().name('year'),
                          self.table.i.month().name('month'),
                          self.table.i.day().name('day')]

        result = to_sql(expr)
        expected = \
            """SELECT extract(`i`, 'year') AS `year`, extract(`i`, 'month') AS `month`,
       extract(`i`, 'day') AS `day`
FROM alltypes"""
        assert result == expected

    def test_timestamp_now(self):
        cases = [
            (ibis.now(), 'now()')
        ]
        self._check_expr_cases(cases)

    def test_timestamp_deltas(self):
        units = ['year', 'month', 'week', 'day',
                 'hour', 'minute', 'second',
                 'millisecond', 'microsecond']

        t = self.table.i
        f = '`i`'

        cases = []
        for unit in units:
            K = 5
            offset = getattr(ibis, unit)(K)
            template = '{0}s_add({1}, {2})'

            cases.append((t + offset, template.format(unit, f, K)))
            cases.append((t - offset, template.format(unit, f, -K)))

        self._check_expr_cases(cases)

    def test_timestamp_literals(self):
        from pandas import Timestamp

        tv1 = '2015-01-01 12:34:56'
        ex1 = ("'2015-01-01 12:34:56'")

        cases = [
            (L(Timestamp(tv1)), ex1),
            (L(Timestamp(tv1).to_pydatetime()), ex1),
            (ibis.timestamp(tv1), ex1)
        ]
        self._check_expr_cases(cases)

    def test_timestamp_from_integer(self):
        col = self.table.c

        cases = [
            (col.to_timestamp(),
             'CAST(from_unixtime(`c`, "yyyy-MM-dd HH:mm:ss") '
             'AS timestamp)'),
            (col.to_timestamp('ms'),
             'CAST(from_unixtime(CAST(`c` / 1000 AS int), '
             '"yyyy-MM-dd HH:mm:ss") '
             'AS timestamp)'),
            (col.to_timestamp('us'),
             'CAST(from_unixtime(CAST(`c` / 1000000 AS int), '
             '"yyyy-MM-dd HH:mm:ss") '
             'AS timestamp)'),
        ]
        self._check_expr_cases(cases)

    def test_correlated_predicate_subquery(self):
        t0 = self.table
        t1 = t0.view()

        expr = t0.g == t1.g

        ctx = ImpalaContext()
        ctx.make_alias(t0)

        # Grab alias from parent context
        subctx = ctx.subcontext()
        subctx.make_alias(t1)
        subctx.make_alias(t0)

        result = self._translate(expr, context=subctx)
        expected = "t0.`g` = t1.`g`"
        assert result == expected

    def test_any_all(self):
        t = self.table

        bool_expr = t.f == 0

        cases = [
            (bool_expr.any(), 'sum(`f` = 0) > 0'),
            (-bool_expr.any(), 'sum(`f` = 0) = 0'),
            (bool_expr.all(), 'sum(`f` = 0) = count(*)'),
            (-bool_expr.all(), 'sum(`f` = 0) < count(*)'),
        ]
        self._check_expr_cases(cases)


class TestUnaryBuiltins(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_numeric_unary_builtins(self):
        # No argument functions
        functions = ['abs', 'ceil', 'floor', 'exp', 'sqrt', 'sign',
                     ('log', 'ln'),
                     ('approx_median', 'appx_median'),
                     ('approx_nunique', 'ndv'),
                     'ln', 'log2', 'log10', 'nullifzero', 'zeroifnull']

        cases = []
        for what in functions:
            if isinstance(what, tuple):
                ibis_name, sql_name = what
            else:
                ibis_name = sql_name = what

            for cname in ['double_col', 'int_col']:
                expr = getattr(self.table[cname], ibis_name)()
                cases.append((expr, '{0}({1})'.format(
                    sql_name, '`{0}`'.format(cname))))

        self._check_expr_cases(cases)

    def test_log_other_bases(self):
        cases = [
            (self.table.double_col.log(5), 'log(`double_col`, 5)')
        ]
        self._check_expr_cases(cases)

    def test_round(self):
        cases = [
            (self.table.double_col.round(), 'round(`double_col`)'),
            (self.table.double_col.round(0), 'round(`double_col`, 0)'),
            (self.table.double_col.round(2, ), 'round(`double_col`, 2)'),
            (self.table.double_col.round(self.table.tinyint_col),
             'round(`double_col`, `tinyint_col`)')
        ]
        self._check_expr_cases(cases)

    def test_hash(self):
        expr = self.table.int_col.hash()
        assert isinstance(expr, ir.Int64Array)
        assert isinstance(self.table.int_col.sum().hash(),
                          ir.Int64Scalar)

        cases = [
            (self.table.int_col.hash(), 'fnv_hash(`int_col`)')
        ]
        self._check_expr_cases(cases)

    def test_reduction_where(self):
        cond = self.table.bigint_col < 70
        c = self.table.double_col
        tmp = ('{0}(CASE WHEN `bigint_col` < 70 THEN `double_col` '
               'ELSE NULL END)')
        cases = [
            (c.sum(where=cond), tmp.format('sum')),
            (c.count(where=cond), tmp.format('count')),
            (c.mean(where=cond), tmp.format('avg')),
            (c.max(where=cond), tmp.format('max')),
            (c.min(where=cond), tmp.format('min')),
            (c.std(where=cond), tmp.format('stddev')),
            (c.std(where=cond, how='pop'), tmp.format('stddev_pop')),
            (c.var(where=cond), tmp.format('variance')),
            (c.var(where=cond, how='pop'), tmp.format('variance_pop')),
        ]
        self._check_expr_cases(cases)

    def test_reduction_invalid_where(self):
        condbad_literal = L('T')
        c = self.table.double_col
        for reduction in [c.sum, c.count, c.mean, c.max, c.min]:
            with self.assertRaises(TypeError):
                reduction(where=condbad_literal)


class TestCaseExprs(unittest.TestCase, ExprSQLTest, ExprTestCases):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_isnull_1_0(self):
        expr = self.table.g.isnull().ifelse(1, 0)

        result = self._translate(expr)
        expected = 'CASE WHEN `g` IS NULL THEN 1 ELSE 0 END'
        assert result == expected

        # inside some other function
        result = self._translate(expr.sum())
        expected = 'sum(CASE WHEN `g` IS NULL THEN 1 ELSE 0 END)'
        assert result == expected

    def test_simple_case(self):
        expr = self._case_simple_case()
        result = self._translate(expr)
        expected = """CASE `g`
  WHEN 'foo' THEN 'bar'
  WHEN 'baz' THEN 'qux'
  ELSE 'default'
END"""
        assert result == expected

    def test_search_case(self):
        expr = self._case_search_case()
        result = self._translate(expr)
        expected = """CASE
  WHEN `f` > 0 THEN `d` * 2
  WHEN `c` < 0 THEN `a` * 2
  ELSE NULL
END"""
        assert result == expected

    def test_where_use_if(self):
        expr = ibis.where(self.table.f > 0, self.table.e, self.table.a)
        assert isinstance(expr, ir.FloatValue)

        result = self._translate(expr)
        expected = "if(`f` > 0, `e`, `a`)"
        assert result == expected

    def test_nullif_ifnull(self):
        table = self.con.table('tpch_lineitem')

        f = table.l_quantity

        cases = [
            (f.nullif(f == 0),
             'nullif(`l_quantity`, `l_quantity` = 0)'),
            (f.fillna(0),
             'isnull(`l_quantity`, CAST(0 AS decimal(12,2)))'),
        ]
        self._check_expr_cases(cases)

    def test_decimal_fillna_cast_arg(self):
        table = self.con.table('tpch_lineitem')
        f = table.l_extendedprice

        cases = [
            (f.fillna(0),
             'isnull(`l_extendedprice`, CAST(0 AS decimal(12,2)))'),
            (f.fillna(0.0), 'isnull(`l_extendedprice`, 0.0)'),
        ]
        self._check_expr_cases(cases)


class TestBucketHistogram(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_bucket_to_case(self):
        buckets = [0, 10, 25, 50]

        expr1 = self.table.f.bucket(buckets)
        expected1 = """\
CASE
  WHEN (`f` >= 0) AND (`f` < 10) THEN 0
  WHEN (`f` >= 10) AND (`f` < 25) THEN 1
  WHEN (`f` >= 25) AND (`f` <= 50) THEN 2
  ELSE NULL
END"""

        expr2 = self.table.f.bucket(buckets, close_extreme=False)
        expected2 = """\
CASE
  WHEN (`f` >= 0) AND (`f` < 10) THEN 0
  WHEN (`f` >= 10) AND (`f` < 25) THEN 1
  WHEN (`f` >= 25) AND (`f` < 50) THEN 2
  ELSE NULL
END"""

        expr3 = self.table.f.bucket(buckets, closed='right')
        expected3 = """\
CASE
  WHEN (`f` >= 0) AND (`f` <= 10) THEN 0
  WHEN (`f` > 10) AND (`f` <= 25) THEN 1
  WHEN (`f` > 25) AND (`f` <= 50) THEN 2
  ELSE NULL
END"""

        expr4 = self.table.f.bucket(buckets, closed='right',
                                    close_extreme=False)
        expected4 = """\
CASE
  WHEN (`f` > 0) AND (`f` <= 10) THEN 0
  WHEN (`f` > 10) AND (`f` <= 25) THEN 1
  WHEN (`f` > 25) AND (`f` <= 50) THEN 2
  ELSE NULL
END"""

        expr5 = self.table.f.bucket(buckets, include_under=True)
        expected5 = """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (`f` >= 0) AND (`f` < 10) THEN 1
  WHEN (`f` >= 10) AND (`f` < 25) THEN 2
  WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
  ELSE NULL
END"""

        expr6 = self.table.f.bucket(buckets,
                                    include_under=True,
                                    include_over=True)
        expected6 = """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (`f` >= 0) AND (`f` < 10) THEN 1
  WHEN (`f` >= 10) AND (`f` < 25) THEN 2
  WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
  WHEN `f` > 50 THEN 4
  ELSE NULL
END"""

        expr7 = self.table.f.bucket(buckets,
                                    close_extreme=False,
                                    include_under=True,
                                    include_over=True)
        expected7 = """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (`f` >= 0) AND (`f` < 10) THEN 1
  WHEN (`f` >= 10) AND (`f` < 25) THEN 2
  WHEN (`f` >= 25) AND (`f` < 50) THEN 3
  WHEN `f` >= 50 THEN 4
  ELSE NULL
END"""

        expr8 = self.table.f.bucket(buckets, closed='right',
                                    close_extreme=False,
                                    include_under=True)
        expected8 = """\
CASE
  WHEN `f` <= 0 THEN 0
  WHEN (`f` > 0) AND (`f` <= 10) THEN 1
  WHEN (`f` > 10) AND (`f` <= 25) THEN 2
  WHEN (`f` > 25) AND (`f` <= 50) THEN 3
  ELSE NULL
END"""

        expr9 = self.table.f.bucket([10], closed='right',
                                    include_over=True,
                                    include_under=True)
        expected9 = """\
CASE
  WHEN `f` <= 10 THEN 0
  WHEN `f` > 10 THEN 1
  ELSE NULL
END"""

        expr10 = self.table.f.bucket([10], include_over=True,
                                     include_under=True)
        expected10 = """\
CASE
  WHEN `f` < 10 THEN 0
  WHEN `f` >= 10 THEN 1
  ELSE NULL
END"""

        cases = [
            (expr1, expected1),
            (expr2, expected2),
            (expr3, expected3),
            (expr4, expected4),
            (expr5, expected5),
            (expr6, expected6),
            (expr7, expected7),
            (expr8, expected8),
            (expr9, expected9),
            (expr10, expected10),
        ]
        self._check_expr_cases(cases)

    def test_cast_category_to_int_noop(self):
        # Because the bucket result is an integer, no explicit cast is
        # necessary
        expr = (self.table.f.bucket([10], include_over=True,
                                    include_under=True)
                .cast('int32'))

        expected = """\
CASE
  WHEN `f` < 10 THEN 0
  WHEN `f` >= 10 THEN 1
  ELSE NULL
END"""

        expr2 = (self.table.f.bucket([10], include_over=True,
                                     include_under=True)
                 .cast('double'))

        expected2 = """\
CAST(CASE
  WHEN `f` < 10 THEN 0
  WHEN `f` >= 10 THEN 1
  ELSE NULL
END AS double)"""

        self._check_expr_cases([(expr, expected),
                                (expr2, expected2)])

    def test_bucket_assign_labels(self):
        buckets = [0, 10, 25, 50]
        bucket = self.table.f.bucket(buckets, include_under=True)

        size = self.table.group_by(bucket.name('tier')).size()
        labelled = size.tier.label(['Under 0', '0 to 10',
                                    '10 to 25', '25 to 50'],
                                   nulls='error').name('tier2')
        expr = size[labelled, size['count']]

        expected = """\
SELECT
  CASE `tier`
    WHEN 0 THEN 'Under 0'
    WHEN 1 THEN '0 to 10'
    WHEN 2 THEN '10 to 25'
    WHEN 3 THEN '25 to 50'
    ELSE 'error'
  END AS `tier2`, `count`
FROM (
  SELECT
    CASE
      WHEN `f` < 0 THEN 0
      WHEN (`f` >= 0) AND (`f` < 10) THEN 1
      WHEN (`f` >= 10) AND (`f` < 25) THEN 2
      WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
      ELSE NULL
    END AS `tier`, count(*) AS `count`
  FROM alltypes
  GROUP BY 1
) t0"""

        result = to_sql(expr)

        assert result == expected

        self.assertRaises(ValueError, size.tier.label, ['a', 'b', 'c'])
        self.assertRaises(ValueError, size.tier.label,
                          ['a', 'b', 'c', 'd', 'e'])


class TestInNotIn(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_field_in_literals(self):
        cases = [
            (self.table.g.isin(["foo", "bar", "baz"]),
             "`g` IN ('foo', 'bar', 'baz')"),
            (self.table.g.notin(["foo", "bar", "baz"]),
             "`g` NOT IN ('foo', 'bar', 'baz')")
        ]
        self._check_expr_cases(cases)

    def test_literal_in_list(self):
        cases = [
            (L(2).isin([self.table.a, self.table.b, self.table.c]),
             '2 IN (`a`, `b`, `c`)'),
            (L(2).notin([self.table.a, self.table.b, self.table.c]),
             '2 NOT IN (`a`, `b`, `c`)')
        ]
        self._check_expr_cases(cases)

    def test_isin_notin_in_select(self):
        filtered = self.table[self.table.g.isin(["foo", "bar"])]
        result = to_sql(filtered)
        expected = """SELECT *
FROM alltypes
WHERE `g` IN ('foo', 'bar')"""
        assert result == expected

        filtered = self.table[self.table.g.notin(["foo", "bar"])]
        result = to_sql(filtered)
        expected = """SELECT *
FROM alltypes
WHERE `g` NOT IN ('foo', 'bar')"""
        assert result == expected


class TestCoalesceGreaterLeast(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_coalesce(self):
        t = self.table
        cases = [
            (ibis.coalesce(t.string_col, 'foo'),
             "coalesce(`string_col`, 'foo')"),
            (ibis.coalesce(t.int_col, t.bigint_col),
             'coalesce(`int_col`, `bigint_col`)'),
        ]
        self._check_expr_cases(cases)

    def test_greatest(self):
        t = self.table
        cases = [
            (ibis.greatest(t.string_col, 'foo'),
             "greatest(`string_col`, 'foo')"),
            (ibis.greatest(t.int_col, t.bigint_col),
             'greatest(`int_col`, `bigint_col`)'),
        ]
        self._check_expr_cases(cases)

    def test_least(self):
        t = self.table
        cases = [
            (ibis.least(t.string_col, 'foo'),
             "least(`string_col`, 'foo')"),
            (ibis.least(t.int_col, t.bigint_col),
             'least(`int_col`, `bigint_col`)'),
        ]
        self._check_expr_cases(cases)


class TestAnalyticFunctions(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_analytic_exprs(self):
        t = self.table

        w = ibis.window(order_by=t.float_col)

        cases = [
            (ibis.row_number().over(w),
             'row_number() OVER (ORDER BY `float_col`) - 1'),
            (t.string_col.lag(), 'lag(`string_col`)'),
            (t.string_col.lag(2), 'lag(`string_col`, 2)'),
            (t.string_col.lag(default=0), 'lag(`string_col`, 1, 0)'),
            (t.string_col.lead(), 'lead(`string_col`)'),
            (t.string_col.lead(2), 'lead(`string_col`, 2)'),
            (t.string_col.lead(default=0), 'lead(`string_col`, 1, 0)'),
            (t.double_col.first(), 'first_value(`double_col`)'),
            (t.double_col.last(), 'last_value(`double_col`)'),
            # (t.double_col.nth(4), 'first_value(lag(double_col, 4 - 1))')
        ]
        self._check_expr_cases(cases)


class TestStringBuiltins(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_unary_ops(self):
        s = self.table.string_col
        cases = [
            (s.lower(), 'lower(`string_col`)'),
            (s.upper(), 'upper(`string_col`)'),
            (s.reverse(), 'reverse(`string_col`)'),
            (s.strip(), 'trim(`string_col`)'),
            (s.lstrip(), 'ltrim(`string_col`)'),
            (s.rstrip(), 'rtrim(`string_col`)'),
            (s.capitalize(), 'initcap(`string_col`)'),
            (s.length(), 'length(`string_col`)'),
            (s.ascii_str(), 'ascii(`string_col`)')
        ]
        self._check_expr_cases(cases)

    def test_substr(self):
        # Database numbers starting from 1
        cases = [
            (self.table.string_col.substr(2), 'substr(`string_col`, 2 + 1)'),
            (self.table.string_col.substr(0, 3),
             'substr(`string_col`, 0 + 1, 3)')
        ]
        self._check_expr_cases(cases)

    def test_strright(self):
        cases = [
            (self.table.string_col.right(4), 'strright(`string_col`, 4)')
        ]
        self._check_expr_cases(cases)

    def test_like(self):
        cases = [
            (self.table.string_col.like('foo%'), "`string_col` LIKE 'foo%'")
        ]
        self._check_expr_cases(cases)

    def test_rlike(self):
        ex = "`string_col` RLIKE '[\d]+'"
        cases = [
            (self.table.string_col.rlike('[\d]+'), ex),
            (self.table.string_col.re_search('[\d]+'), ex),
        ]
        self._check_expr_cases(cases)

    def test_re_extract(self):
        sql = "regexp_extract(`string_col`, '[\d]+', 0)"
        cases = [
            (self.table.string_col.re_extract('[\d]+', 0), sql)
        ]
        self._check_expr_cases(cases)

    def test_re_replace(self):
        sql = "regexp_replace(`string_col`, '[\d]+', 'aaa')"
        cases = [
            (self.table.string_col.re_replace('[\d]+', 'aaa'), sql)
        ]
        self._check_expr_cases(cases)

    def test_parse_url(self):
        sql = "parse_url(`string_col`, 'HOST')"
        cases = [
            (self.table.string_col.parse_url('HOST'), sql)
        ]
        self._check_expr_cases(cases)

    def test_repeat(self):
        cases = [
            (self.table.string_col.repeat(2), 'repeat(`string_col`, 2)')
        ]
        self._check_expr_cases(cases)

    def test_translate(self):
        cases = [
            (self.table.string_col.translate('a', 'b'),
             "translate(`string_col`, 'a', 'b')")
        ]
        self._check_expr_cases(cases)

    def test_find(self):
        s = self.table.string_col
        i1 = self.table.tinyint_col
        cases = [
            (s.find('a'), "locate('a', `string_col`) - 1"),
            (s.find('a', 2), "locate('a', `string_col`, 3) - 1"),
            (s.find('a', start=i1),
             "locate('a', `string_col`, `tinyint_col` + 1) - 1")
        ]
        self._check_expr_cases(cases)

    def test_lpad(self):
        cases = [
            (self.table.string_col.lpad(1, 'a'), "lpad(`string_col`, 1, 'a')"),
            (self.table.string_col.lpad(25), "lpad(`string_col`, 25, ' ')")
        ]
        self._check_expr_cases(cases)

    def test_rpad(self):
        cases = [
            (self.table.string_col.rpad(1, 'a'), "rpad(`string_col`, 1, 'a')"),
            (self.table.string_col.rpad(25), "rpad(`string_col`, 25, ' ')")
        ]
        self._check_expr_cases(cases)

    def test_find_in_set(self):
        cases = [
            (self.table.string_col.find_in_set(['a']),
             "find_in_set(`string_col`, 'a') - 1"),
            (self.table.string_col.find_in_set(['a', 'b']),
             "find_in_set(`string_col`, 'a,b') - 1")
        ]
        self._check_expr_cases(cases)

    def test_string_join(self):
        cases = [
            (L(',').join(['a', 'b']), "concat_ws(',', 'a', 'b')")
        ]
        self._check_expr_cases(cases)


class TestImpalaExprs(ImpalaE2E, unittest.TestCase, ExprTestCases):

    def test_embedded_identifier_quoting(self):
        t = self.con.table('functional_alltypes')

        expr = (t[[(t.double_col * 2).name('double(fun)')]]
                ['double(fun)'].sum())
        expr.execute()

    def test_table_info(self):
        t = self.con.table('functional_alltypes')
        buf = StringIO()
        t.info(buf=buf)

        assert buf.getvalue() is not None

    def test_execute_exprs_no_table_ref(self):
        cases = [
            (L(1) + L(2), 3)
        ]

        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected

        # ExprList
        exlist = ibis.api.expr_list([L(1).name('a'),
                                     ibis.now().name('b'),
                                     L(2).log().name('c')])
        self.con.execute(exlist)

    def test_summary_execute(self):
        table = self.alltypes

        # also test set_column while we're at it
        table = table.set_column('double_col',
                                 table.double_col * 2)

        expr = table.double_col.summary()
        repr(expr)

        result = expr.execute()
        assert isinstance(result, pd.DataFrame)

        expr = (table.group_by('string_col')
                .aggregate([table.double_col.summary().prefix('double_'),
                            table.float_col.summary().prefix('float_'),
                            table.string_col.summary().suffix('_string')]))
        result = expr.execute()
        assert isinstance(result, pd.DataFrame)

    def test_distinct_array(self):
        table = self.alltypes

        expr = table.string_col.distinct()
        result = self.con.execute(expr)
        assert isinstance(result, pd.Series)

    def test_decimal_metadata(self):
        table = self.con.table('tpch_lineitem')

        expr = table.l_quantity
        assert expr._precision == 12
        assert expr._scale == 2

        # TODO: what if user impyla version does not have decimal Metadata?

    def test_builtins_1(self):
        table = self.alltypes

        i1 = table.tinyint_col
        i4 = table.int_col
        i8 = table.bigint_col
        d = table.double_col
        s = table.string_col

        exprs = [
            api.now(),
            api.e,

            # hash functions
            i4.hash(),
            d.hash(),
            s.hash(),

            # modulus cases
            i1 % 5,
            i4 % 10,
            20 % i1,
            d % 5,

            i1.zeroifnull(),
            i4.zeroifnull(),
            i8.zeroifnull(),

            i4.to_timestamp('s'),
            i4.to_timestamp('ms'),
            i4.to_timestamp('us'),

            i8.to_timestamp(),

            d.abs(),
            d.cast('decimal(12, 2)'),
            d.cast('int32'),
            d.ceil(),
            d.exp(),
            d.isnull(),
            d.fillna(0),
            d.floor(),
            d.log(),
            d.ln(),
            d.log2(),
            d.log10(),
            d.notnull(),

            d.zeroifnull(),
            d.nullifzero(),

            d.round(),
            d.round(2),
            d.round(i1),

            i1.sign(),
            i4.sign(),
            d.sign(),

            # conv
            i1.convert_base(10, 2),
            i4.convert_base(10, 2),
            i8.convert_base(10, 2),
            s.convert_base(10, 2),

            d.sqrt(),
            d.zeroifnull(),

            # nullif cases
            5 / i1.nullif(0),
            5 / i1.nullif(i4),
            5 / i4.nullif(0),
            5 / d.nullif(0),

            api.literal(5).isin([i1, i4, d]),

            # tier and histogram
            d.bucket([0, 10, 25, 50, 100]),
            d.bucket([0, 10, 25, 50], include_over=True),
            d.bucket([0, 10, 25, 50], include_over=True, close_extreme=False),
            d.bucket([10, 25, 50, 100], include_under=True),

            d.histogram(10),
            d.histogram(5, base=10),
            d.histogram(base=10, binwidth=5),

            # coalesce-like cases
            api.coalesce(table.int_col,
                         api.null(),
                         table.smallint_col,
                         table.bigint_col, 5),
            api.greatest(table.float_col,
                         table.double_col, 5),
            api.least(table.string_col, 'foo'),

            # string stuff
            s.contains('6'),
            s.like('6%'),
            s.re_search('[\d]+'),
            s.re_extract('[\d]+', 0),
            s.re_replace('[\d]+', 'a'),
            s.repeat(2),
            s.translate("a", "b"),
            s.find("a"),
            s.lpad(10, 'a'),
            s.rpad(10, 'a'),
            s.find_in_set(["a"]),
            s.lower(),
            s.upper(),
            s.reverse(),
            s.ascii_str(),
            s.length(),
            s.strip(),
            s.lstrip(),
            s.strip(),

            # strings with int expr inputs
            s.left(i1),
            s.right(i1),
            s.substr(i1, i1 + 2),
            s.repeat(i1)
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        projection = table[proj_exprs]
        projection.limit(10).execute()

        self._check_impala_output_types_match(projection)

    def _check_impala_output_types_match(self, table):
        query = to_sql(table)
        t = self.con.sql(query)

        def _clean_type(x):
            if isinstance(x, Category):
                x = x.to_integer_type()
            return x

        left, right = t.schema(), table.schema()
        for i, (n, l, r) in enumerate(zip(left.names, left.types,
                                          right.types)):
            l = _clean_type(l)
            r = _clean_type(r)

            if l != r:
                pytest.fail('Value for {0} had left type {1}'
                            ' and right type {2}'.format(n, l, r))

    def assert_cases_equality(self, cases):
        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected, to_sql(expr)

    def test_int_builtins(self):
        i8 = L(50)
        i32 = L(50000)

        mod_cases = [
            (i8 % 5, 0),
            (i32 % 10, 0),
            (250 % i8, 0),
        ]

        nullif_cases = [
            (5 / i8.nullif(0), 0.1),
            (5 / i8.nullif(i32), 0.1),
            (5 / i32.nullif(0), 0.0001),
            (i32.zeroifnull(), 50000),
        ]

        self.assert_cases_equality(mod_cases + nullif_cases)

    def test_column_types(self):
        df = self.alltypes.execute()
        assert df.tinyint_col.dtype.name == 'int8'
        assert df.smallint_col.dtype.name == 'int16'
        assert df.int_col.dtype.name == 'int32'
        assert df.bigint_col.dtype.name == 'int64'
        assert df.float_col.dtype.name == 'float32'
        assert df.double_col.dtype.name == 'float64'
        assert pd.core.common.is_datetime64_dtype(df.timestamp_col.dtype)

    def test_timestamp_builtins(self):
        i32 = L(50000)
        i64 = L(5 * 10 ** 8)

        stamp = ibis.timestamp('2009-05-17 12:34:56')

        timestamp_cases = [
            (i32.to_timestamp('s'), pd.to_datetime(50000, unit='s')),
            (i32.to_timestamp('ms'), pd.to_datetime(50000, unit='ms')),
            (i64.to_timestamp(), pd.to_datetime(5 * 10 ** 8, unit='s')),

            (stamp.truncate('y'), pd.Timestamp('2009-01-01')),
            (stamp.truncate('m'), pd.Timestamp('2009-05-01')),
            (stamp.truncate('d'), pd.Timestamp('2009-05-17')),
            (stamp.truncate('h'), pd.Timestamp('2009-05-17 12:00')),
            (stamp.truncate('minute'), pd.Timestamp('2009-05-17 12:34'))
        ]

        self.assert_cases_equality(timestamp_cases)

    def test_decimal_builtins(self):
        d = L(5.245)
        general_cases = [
            (L(-5).abs(), 5),
            (d.cast('int32'), 5),
            (d.ceil(), 6),
            (d.isnull(), False),
            (d.floor(), 5),
            (d.notnull(), True),
            (d.round(), 5),
            (d.round(2), Decimal('5.25')),
            (d.sign(), 1),
        ]
        self.assert_cases_equality(general_cases)

    def test_decimal_builtins_2(self):
        d = L('5.245')
        dc = d.cast('decimal(12,5)')
        cases = [
            (dc % 5, Decimal('0.245')),

            (dc.fillna(0), Decimal('5.245')),

            (dc.exp(), 189.6158),
            (dc.log(), 1.65728),
            (dc.log2(), 2.39094),
            (dc.log10(), 0.71975),
            (dc.sqrt(), 2.29019),
            (dc.zeroifnull(), Decimal('5.245')),
            (-dc, Decimal('-5.245'))
        ]

        for expr, expected in cases:
            result = self.con.execute(expr)
            if isinstance(expected, Decimal):
                tol = Decimal('0.0001')
            else:
                tol = 0.0001
            approx_equal(result, expected, tol)

    def test_string_functions(self):
        string = L('abcd')
        strip_string = L('   a   ')

        cases = [
            (string.length(), 4),
            (L('ABCD').lower(), 'abcd'),
            (string.upper(), 'ABCD'),
            (string.reverse(), 'dcba'),
            (string.ascii_str(), 97),
            (strip_string.strip(), 'a'),
            (strip_string.lstrip(), 'a   '),
            (strip_string.rstrip(), '   a'),
            (string.capitalize(), 'Abcd'),
            (string.substr(0, 2), 'ab'),
            (string.left(2), 'ab'),
            (string.right(2), 'cd'),
            (string.repeat(2), 'abcdabcd'),

            # global replace not available in Impala yet
            # (L('aabbaabbaa').replace('bb', 'B'), 'aaBaaBaa'),

            (L('0123').translate('012', 'abc'), 'abc3'),
            (string.find('a'), 0),
            (L('baaaab').find('b', 2), 5),
            (string.lpad(1, '-'), 'a'),
            (string.lpad(5), ' abcd'),
            (string.rpad(1, '-'), 'a'),
            (string.rpad(5), 'abcd '),
            (string.find_in_set(['a', 'b', 'abcd']), 2),
            (L(', ').join(['a', 'b']), 'a, b'),
            (string.like('a%'), True),
            (string.re_search('[a-z]'), True),

            (string.re_extract('[a-z]', 0), 'a'),
            (string.re_replace('(b)', '2'), 'a2cd'),
        ]

        self._check_cases(cases)

    def _check_cases(self, cases):
        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected

    def test_parse_url(self):
        cases = [
            (L("https://www.cloudera.com").parse_url('HOST'),
             "www.cloudera.com"),

            (L('https://www.youtube.com/watch?v=kEuEcWfewf8&t=10')
             .parse_url('QUERY', 'v'),
             'kEuEcWfewf8'),
        ]
        self._check_cases(cases)

    def test_div_floordiv(self):
        cases = [
            (L(7) / 2, 3.5),
            (L(7) // 2, 3),
            (L(7).floordiv(2), 3),
            (L(2).rfloordiv(7), 3),
        ]

        for expr, expected in cases:
            result = self.con.execute(expr)
            assert result == expected

    def test_filter_predicates(self):
        t = self.con.table('tpch_nation')

        predicates = [
            lambda x: x.n_name.lower().like('%ge%'),
            lambda x: x.n_name.lower().contains('ge'),
            lambda x: x.n_name.lower().rlike('.*ge.*')
        ]

        expr = t
        for pred in predicates:
            expr = expr[pred(expr)].projection([expr])

        expr.execute()

    def test_histogram_value_counts(self):
        t = self.alltypes
        expr = t.double_col.histogram(10).value_counts()
        expr.execute()

    def test_casted_expr_impala_bug(self):
        # Per GH #396. Prior to Impala 2.3.0, there was a bug in the query
        # planner that caused this expression to fail
        expr = self.alltypes.string_col.cast('double').value_counts()
        expr.execute()

    def test_decimal_timestamp_builtins(self):
        table = self.con.table('tpch_lineitem')

        dc = table.l_quantity
        ts = table.l_receiptdate.cast('timestamp')

        exprs = [
            dc % 10,
            dc + 5,
            dc + dc,
            dc / 2,
            dc * 2,
            dc ** 2,
            dc.cast('double'),

            api.where(table.l_discount > 0,
                      dc * table.l_discount, api.NA),

            dc.fillna(0),

            ts < (ibis.now() + ibis.month(3)),
            ts < (ibis.timestamp('2005-01-01') + ibis.month(3)),

            # hashing
            dc.hash(),
            ts.hash(),

            # truncate
            ts.truncate('y'),
            ts.truncate('q'),
            ts.truncate('month'),
            ts.truncate('d'),
            ts.truncate('w'),
            ts.truncate('h'),
            ts.truncate('minute'),
        ]

        timestamp_fields = ['year', 'month', 'day', 'hour', 'minute',
                            'second', 'millisecond', 'microsecond',
                            'week']
        for field in timestamp_fields:
            if hasattr(ts, field):
                exprs.append(getattr(ts, field)())

            offset = getattr(ibis, field)(2)
            exprs.append(ts + offset)
            exprs.append(ts - offset)

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        projection = table[proj_exprs].limit(10)
        projection.execute()

    def test_timestamp_scalar_in_filter(self):
        # #310
        table = self.alltypes

        expr = (table.filter([table.timestamp_col <
                             (ibis.timestamp('2010-01-01') + ibis.month(3)),
                             table.timestamp_col < (ibis.now() + ibis.day(10))
                              ])
                .count())
        expr.execute()

    def test_aggregations(self):
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
            s.approx_nunique(),
            d.approx_median(),
            s.group_concat(),

            d.std(),
            d.std(how='pop'),
            d.var(),
            d.var(how='pop'),

            table.bool_col.any(),
            table.bool_col.notany(),
            -table.bool_col.any(),

            table.bool_col.all(),
            table.bool_col.notall(),
            -table.bool_col.all(),

            table.bool_col.count(where=cond),
            d.sum(where=cond),
            d.mean(where=cond),
            d.min(where=cond),
            d.max(where=cond),
            d.std(where=cond),
            d.var(where=cond),
        ]

        agg_exprs = [expr.name('e%d' % i)
                     for i, expr in enumerate(exprs)]

        agged_table = table.aggregate(agg_exprs)
        agged_table.execute()

    def test_analytic_functions(self):
        t = self.alltypes.limit(1000)

        g = t.group_by('string_col').order_by('double_col')
        f = t.float_col

        exprs = [
            f.lag(),
            f.lead(),
            f.rank(),
            f.dense_rank(),

            f.first(),
            f.last(),

            f.first().over(ibis.window(preceding=10)),
            f.first().over(ibis.window(following=10)),

            ibis.row_number(),
            f.cumsum(),
            f.cummean(),
            f.cummin(),
            f.cummax(),

            # boolean cumulative reductions
            (f == 0).cumany(),
            (f == 0).cumall(),

            f.sum(),
            f.mean(),
            f.min(),
            f.max()
        ]

        proj_exprs = [expr.name('e%d' % i)
                      for i, expr in enumerate(exprs)]

        proj_table = g.mutate(proj_exprs)
        proj_table.execute()

    def test_anti_join_self_reference_works(self):
        case = self._case_self_reference_limit_exists()
        self.con.explain(case)

    def test_tpch_self_join_failure(self):
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')
        orders = self.con.table('tpch_orders')

        fields_of_interest = [
            region.r_name.name('region'),
            nation.n_name.name('nation'),
            orders.o_totalprice.name('amount'),
            orders.o_orderdate.cast('timestamp').name('odate')]

        joined_all = (
            region.join(nation, region.r_regionkey == nation.n_regionkey)
            .join(customer, customer.c_nationkey == nation.n_nationkey)
            .join(orders, orders.o_custkey == customer.c_custkey)
            [fields_of_interest])

        year = joined_all.odate.year().name('year')
        total = joined_all.amount.sum().cast('double').name('total')
        annual_amounts = (joined_all
                          .group_by(['region', year])
                          .aggregate(total))

        current = annual_amounts
        prior = annual_amounts.view()

        yoy_change = (current.total - prior.total).name('yoy_change')
        yoy = (current.join(prior, ((current.region == prior.region) &
                                    (current.year == (prior.year - 1))))
               [current.region, current.year, yoy_change])

        # no analysis failure
        self.con.explain(yoy)

    def test_tpch_correlated_subquery_failure(self):
        # #183 and other issues
        region = self.con.table('tpch_region')
        nation = self.con.table('tpch_nation')
        customer = self.con.table('tpch_customer')
        orders = self.con.table('tpch_orders')

        fields_of_interest = [customer,
                              region.r_name.name('region'),
                              orders.o_totalprice.name('amount'),
                              orders.o_orderdate
                              .cast('timestamp').name('odate')]

        tpch = (region.join(nation, region.r_regionkey == nation.n_regionkey)
                .join(customer, customer.c_nationkey == nation.n_nationkey)
                .join(orders, orders.o_custkey == customer.c_custkey)
                [fields_of_interest])

        t2 = tpch.view()
        conditional_avg = t2[(t2.region == tpch.region)].amount.mean()
        amount_filter = tpch.amount > conditional_avg

        expr = tpch[amount_filter].limit(0)
        self.con.explain(expr)

    def test_non_equijoin(self):
        t = self.con.table('functional_alltypes').limit(100)
        t2 = t.view()

        expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()

        # it works
        expr.execute()

    def test_char_varchar_types(self):
        sql = """\
SELECT CAST(string_col AS varchar(20)) AS varchar_col,
       CAST(string_col AS CHAR(5)) AS char_col
FROM functional_alltypes"""

        t = self.con.sql(sql)

        assert isinstance(t.varchar_col, api.StringArray)
        assert isinstance(t.char_col, api.StringArray)

    def test_unions_with_ctes(self):
        t = self.con.table('functional_alltypes')

        expr1 = (t.group_by(['tinyint_col', 'string_col'])
                 .aggregate(t.double_col.sum().name('metric')))
        expr2 = expr1.view()

        join1 = (expr1.join(expr2, expr1.string_col == expr2.string_col)
                 [[expr1]])
        join2 = join1.view()

        expr = join1.union(join2)
        self.con.explain(expr)
