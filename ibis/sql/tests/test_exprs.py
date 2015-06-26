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

import pandas as pd

from ibis.sql.exprs import ExprTranslator
from ibis.sql.compiler import QueryContext, to_sql
from ibis.expr.tests.mocks import MockConnection
from ibis.compat import unittest
import ibis.expr.types as ir
import ibis


class ExprSQLTest(object):

    def _check_expr_cases(self, cases, context=None, named=False):
        for expr, expected in cases:
            repr(expr)
            result = self._translate(expr, named=named, context=context)
            assert result == expected

    def _translate(self, expr, named=False, context=None):
        translator = ExprTranslator(expr, context=context, named=named)
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
            lit_expr = ibis.literal(value)
            result = self._translate(lit_expr)
            assert result == expected

    def test_string_literals(self):
        cases = [
            ('simple', "'simple'"),
            ('I can\'t', "'I can\\'t'"),
            ('An "escape"', "'An \"escape\"'")
        ]

        for value, expected in cases:
            lit_expr = ibis.literal(value)
            result = self._translate(lit_expr)
            assert result == expected

    def test_decimal_builtins(self):
        t = self.con.table('tpch_lineitem')
        col = t.l_extendedprice
        cases = [
            (col.precision(), 'precision(l_extendedprice)'),
            (col.scale(), 'scale(l_extendedprice)'),
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
        context = QueryContext()

        table1 = ibis.table([
            ('key1', 'string'),
            ('value1', 'double')
        ])

        table2 = ibis.table([
            ('key2', 'string'),
            ('value and2', 'double')
        ])

        context.set_alias(table1, 't0')
        context.set_alias(table2, 't1')

        expr = table1['value1'] - table2['value and2']

        result = self._translate(expr, context=context)
        expected = 't0.value1 - t1.`value and2`'
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
            (g.cast('double').name('g_dub'), 'CAST(g AS double) AS `g_dub`'),
            (g.name('has a space'), 'g AS `has a space`'),
            (((a - b) * a).name('expr'), '(a - b) * a AS `expr`')
        ]

        return self._check_expr_cases(cases, named=True)

    def test_binary_infix_operators(self):
        # For each function, verify that the generated code is what we expect
        a, b, h = self.table.get_columns(['a', 'b', 'h'])
        bool_col = a > 0

        cases = [
            (a + b, 'a + b'),
            (a - b, 'a - b'),
            (a * b, 'a * b'),
            (a / b, 'a / b'),
            (a ** b, 'pow(a, b)'),
            (a < b, 'a < b'),
            (a <= b, 'a <= b'),
            (a > b, 'a > b'),
            (a >= b, 'a >= b'),
            (a == b, 'a = b'),
            (a != b, 'a != b'),
            (h & bool_col, 'h AND (a > 0)'),
            (h | bool_col, 'h OR (a > 0)'),
            # xor is brute force
            (h ^ bool_col, '(h OR (a > 0)) AND NOT (h AND (a > 0))')
        ]
        self._check_expr_cases(cases)

    def test_binary_infix_parenthesization(self):
        a, b, c = self.table.get_columns(['a', 'b', 'c'])

        cases = [
            ((a + b) + c, '(a + b) + c'),
            (a.log() + c, 'ln(a) + c'),
            (b + (-(a + c)), 'b + (-(a + c))')
        ]

        self._check_expr_cases(cases)

    def test_between(self):
        cases = [
            (self.table.f.between(0, 1), 'f BETWEEN 0 AND 1')
        ]
        self._check_expr_cases(cases)

    def test_isnull_notnull(self):
        cases = [
            (self.table['g'].isnull(), 'g IS NULL'),
            (self.table['a'].notnull(), 'a IS NOT NULL'),
            ((self.table['a'] + self.table['b']).isnull(), 'a + b IS NULL')
        ]
        self._check_expr_cases(cases)

    def test_casts(self):
        a, d, g = self.table.get_columns(['a', 'd', 'g'])
        cases = [
            (a.cast('int16'), 'CAST(a AS smallint)'),
            (a.cast('int32'), 'CAST(a AS int)'),
            (a.cast('int64'), 'CAST(a AS bigint)'),
            (a.cast('float'), 'CAST(a AS float)'),
            (a.cast('double'), 'CAST(a AS double)'),
            (a.cast('string'), 'CAST(a AS string)'),
            (d.cast('int8'), 'CAST(d AS tinyint)'),
            (g.cast('double'), 'CAST(g AS double)'),
            (g.cast('timestamp'), 'CAST(g AS timestamp)')
        ]
        self._check_expr_cases(cases)

    def test_misc_conditionals(self):
        a = self.table.a
        cases = [
            (a.nullif(0), 'nullif(a, 0)')
        ]
        self._check_expr_cases(cases)

    def test_decimal_casts(self):
        cases = [
            (ibis.literal('9.9999999').cast('decimal(38,5)'),
             "CAST('9.9999999' AS decimal(38,5))"),
            (self.table.f.cast('decimal(12,2)'), "CAST(f AS decimal(12,2))")
        ]
        self._check_expr_cases(cases)

    def test_negate(self):
        cases = [
            (-self.table['a'], '-a'),
            (-self.table['f'], '-f'),
            (-self.table['h'], 'NOT h')
        ]
        self._check_expr_cases(cases)

    def test_timestamp_extract_field(self):
        fields = ['year', 'month', 'day', 'hour', 'minute',
                  'second', 'millisecond']

        cases = [(getattr(self.table.i, field)(),
                  "extract(i, '{0}')".format(field))
                 for field in fields]
        self._check_expr_cases(cases)

        # integration with SQL translation
        expr = self.table[self.table.i.year().name('year'),
                          self.table.i.month().name('month'),
                          self.table.i.day().name('day')]

        result = to_sql(expr)
        expected = \
            """SELECT extract(i, 'year') AS `year`, extract(i, 'month') AS `month`,
       extract(i, 'day') AS `day`
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
        f = 'i'

        cases = []
        for unit in units:
            K = 5
            offset = getattr(ibis, unit)(K)
            template = '{0}s_add({1}, {2})'

            cases.append((t + offset, template.format(unit, f, K)))
            cases.append((t - offset, template.format(unit, f, -K)))

        self._check_expr_cases(cases)

    def test_timestamp_literals(self):
        tv1 = '2015-01-01 12:34:56'
        ex1 = ("'2015-01-01 12:34:56'")

        cases = [
            (ibis.literal(pd.Timestamp(tv1)), ex1),
            (ibis.literal(pd.Timestamp(tv1).to_pydatetime()), ex1),
            (ibis.timestamp(tv1), ex1)
        ]
        self._check_expr_cases(cases)

    def test_timestamp_from_integer(self):
        col = self.table.c

        cases = [
            (col.to_timestamp(),
             'CAST(from_unixtime(c, "yyyy-MM-dd HH:mm:ss") '
             'AS timestamp)'),
            (col.to_timestamp('ms'),
             'CAST(from_unixtime(CAST(c / 1000 AS int), '
             '"yyyy-MM-dd HH:mm:ss") '
             'AS timestamp)'),
            (col.to_timestamp('us'),
             'CAST(from_unixtime(CAST(c / 1000000 AS int), '
             '"yyyy-MM-dd HH:mm:ss") '
             'AS timestamp)'),
        ]
        self._check_expr_cases(cases)

    def test_correlated_predicate_subquery(self):
        t0 = self.table
        t1 = t0.view()

        expr = t0.g == t1.g

        ctx = QueryContext()
        ctx.make_alias(t0)

        # Grab alias from parent context
        subctx = ctx.subcontext()
        subctx.make_alias(t1)
        subctx.make_alias(t0)

        result = self._translate(expr, context=subctx)
        expected = "t0.g = t1.g"
        assert result == expected


class TestUnaryBuiltins(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_numeric_monadic_builtins(self):
        # No argument functions
        functions = ['abs', 'ceil', 'floor', 'exp', 'sqrt', 'sign',
                     ('log', 'ln'),
                     ('approx_median', 'appx_median'),
                     ('approx_nunique', 'ndv'),
                     'ln', 'log2', 'log10', 'zeroifnull']

        cases = []
        for what in functions:
            if isinstance(what, tuple):
                ibis_name, sql_name = what
            else:
                ibis_name = sql_name = what

            for cname in ['double_col', 'int_col']:
                expr = getattr(self.table[cname], ibis_name)()
                cases.append((expr, '{0}({1})'.format(sql_name, cname)))

        self._check_expr_cases(cases)

    def test_log_other_bases(self):
        cases = [
            (self.table.double_col.log(5), 'log(double_col, 5)')
        ]
        self._check_expr_cases(cases)

    def test_round(self):
        cases = [
            (self.table.double_col.round(), 'round(double_col)'),
            (self.table.double_col.round(0), 'round(double_col, 0)'),
            (self.table.double_col.round(2, ), 'round(double_col, 2)'),
            (self.table.double_col.round(self.table.tinyint_col),
             'round(double_col, tinyint_col)')
        ]
        self._check_expr_cases(cases)

    def test_hash(self):
        expr = self.table.int_col.hash()
        assert isinstance(expr, ir.Int64Array)
        assert isinstance(self.table.int_col.sum().hash(),
                          ir.Int64Scalar)

        cases = [
            (self.table.int_col.hash(), 'fnv_hash(int_col)')
        ]
        self._check_expr_cases(cases)

    def test_reduction_where(self):
        cond = self.table.bigint_col < 70
        c = self.table.double_col
        tmp = '{0}(CASE WHEN bigint_col < 70 THEN double_col ELSE NULL END)'
        cases = [
            (c.sum(where=cond), tmp.format('sum')),
            (c.count(where=cond), tmp.format('count')),
            (c.mean(where=cond), tmp.format('avg')),
            (c.max(where=cond), tmp.format('max')),
            (c.min(where=cond), tmp.format('min')),
        ]
        self._check_expr_cases(cases)

    def test_reduction_invalid_where(self):
        condbad_literal = ibis.literal('T')
        c = self.table.double_col
        for reduction in [c.sum, c.count, c.mean, c.max, c.min]:
            with self.assertRaises(TypeError):
                reduction(where=condbad_literal)


class TestCaseExprs(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_isnull_1_0(self):
        expr = self.table.g.isnull().ifelse(1, 0)

        result = self._translate(expr)
        expected = 'CASE WHEN g IS NULL THEN 1 ELSE 0 END'
        assert result == expected

        # inside some other function
        result = self._translate(expr.sum())
        expected = 'sum(CASE WHEN g IS NULL THEN 1 ELSE 0 END)'
        assert result == expected

    def test_simple_case(self):
        expr = (self.table.g.case()
                .when('foo', 'bar')
                .when('baz', 'qux')
                .else_('default')
                .end())

        result = self._translate(expr)
        expected = """CASE g
  WHEN 'foo' THEN 'bar'
  WHEN 'baz' THEN 'qux'
  ELSE 'default'
END"""
        assert result == expected

    def test_search_case(self):
        expr = (ibis.case()
                .when(self.table.f > 0, self.table.d * 2)
                .when(self.table.c < 0, self.table.a * 2)
                .end())

        result = self._translate(expr)
        expected = """CASE
  WHEN f > 0 THEN d * 2
  WHEN c < 0 THEN a * 2
  ELSE NULL
END"""
        assert result == expected

    def test_where_use_if(self):
        expr = ibis.where(self.table.f > 0, self.table.e, self.table.a)
        assert isinstance(expr, ir.FloatValue)

        result = self._translate(expr)
        expected = "if(f > 0, e, a)"
        assert result == expected

    def test_nullif_ifnull(self):
        table = self.con.table('tpch_lineitem')

        f = table.l_quantity

        cases = [
            (f.nullif(f == 0),
             'nullif(l_quantity, l_quantity = 0)'),
            (f.fillna(0),
             'isnull(l_quantity, CAST(0 AS decimal(12,2)))'),
        ]
        self._check_expr_cases(cases)

    def test_decimal_fillna_cast_arg(self):
        table = self.con.table('tpch_lineitem')
        f = table.l_extendedprice

        cases = [
            (f.fillna(0),
             'isnull(l_extendedprice, CAST(0 AS decimal(12,2)))')
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
  WHEN (f >= 0) AND (f < 10) THEN 0
  WHEN (f >= 10) AND (f < 25) THEN 1
  WHEN (f >= 25) AND (f <= 50) THEN 2
  ELSE NULL
END"""

        expr2 = self.table.f.bucket(buckets, close_extreme=False)
        expected2 = """\
CASE
  WHEN (f >= 0) AND (f < 10) THEN 0
  WHEN (f >= 10) AND (f < 25) THEN 1
  WHEN (f >= 25) AND (f < 50) THEN 2
  ELSE NULL
END"""

        expr3 = self.table.f.bucket(buckets, closed='right')
        expected3 = """\
CASE
  WHEN (f >= 0) AND (f <= 10) THEN 0
  WHEN (f > 10) AND (f <= 25) THEN 1
  WHEN (f > 25) AND (f <= 50) THEN 2
  ELSE NULL
END"""

        expr4 = self.table.f.bucket(buckets, closed='right',
                                    close_extreme=False)
        expected4 = """\
CASE
  WHEN (f > 0) AND (f <= 10) THEN 0
  WHEN (f > 10) AND (f <= 25) THEN 1
  WHEN (f > 25) AND (f <= 50) THEN 2
  ELSE NULL
END"""

        expr5 = self.table.f.bucket(buckets, include_under=True)
        expected5 = """\
CASE
  WHEN f < 0 THEN 0
  WHEN (f >= 0) AND (f < 10) THEN 1
  WHEN (f >= 10) AND (f < 25) THEN 2
  WHEN (f >= 25) AND (f <= 50) THEN 3
  ELSE NULL
END"""

        expr6 = self.table.f.bucket(buckets,
                                    include_under=True,
                                    include_over=True)
        expected6 = """\
CASE
  WHEN f < 0 THEN 0
  WHEN (f >= 0) AND (f < 10) THEN 1
  WHEN (f >= 10) AND (f < 25) THEN 2
  WHEN (f >= 25) AND (f <= 50) THEN 3
  WHEN f > 50 THEN 4
  ELSE NULL
END"""

        expr7 = self.table.f.bucket(buckets,
                                    close_extreme=False,
                                    include_under=True,
                                    include_over=True)
        expected7 = """\
CASE
  WHEN f < 0 THEN 0
  WHEN (f >= 0) AND (f < 10) THEN 1
  WHEN (f >= 10) AND (f < 25) THEN 2
  WHEN (f >= 25) AND (f < 50) THEN 3
  WHEN f >= 50 THEN 4
  ELSE NULL
END"""

        expr8 = self.table.f.bucket(buckets, closed='right',
                                    close_extreme=False,
                                    include_under=True)
        expected8 = """\
CASE
  WHEN f <= 0 THEN 0
  WHEN (f > 0) AND (f <= 10) THEN 1
  WHEN (f > 10) AND (f <= 25) THEN 2
  WHEN (f > 25) AND (f <= 50) THEN 3
  ELSE NULL
END"""

        expr9 = self.table.f.bucket([10], closed='right',
                                    include_over=True,
                                    include_under=True)
        expected9 = """\
CASE
  WHEN f <= 10 THEN 0
  WHEN f > 10 THEN 1
  ELSE NULL
END"""

        expr10 = self.table.f.bucket([10], include_over=True,
                                     include_under=True)
        expected10 = """\
CASE
  WHEN f < 10 THEN 0
  WHEN f >= 10 THEN 1
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
  WHEN f < 10 THEN 0
  WHEN f >= 10 THEN 1
  ELSE NULL
END"""

        expr2 = (self.table.f.bucket([10], include_over=True,
                                     include_under=True)
                 .cast('double'))

        expected2 = """\
CAST(CASE
  WHEN f < 10 THEN 0
  WHEN f >= 10 THEN 1
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
  CASE tier
    WHEN 0 THEN 'Under 0'
    WHEN 1 THEN '0 to 10'
    WHEN 2 THEN '10 to 25'
    WHEN 3 THEN '25 to 50'
    ELSE 'error'
  END AS `tier2`, count
FROM (
  SELECT
    CASE
      WHEN f < 0 THEN 0
      WHEN (f >= 0) AND (f < 10) THEN 1
      WHEN (f >= 10) AND (f < 25) THEN 2
      WHEN (f >= 25) AND (f <= 50) THEN 3
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
             "g IN ('foo', 'bar', 'baz')"),
            (self.table.g.notin(["foo", "bar", "baz"]),
             "g NOT IN ('foo', 'bar', 'baz')")
        ]
        self._check_expr_cases(cases)

    def test_literal_in_list(self):
        cases = [
            (ibis.literal(2).isin([self.table.a, self.table.b, self.table.c]),
             '2 IN (a, b, c)'),
            (ibis.literal(2).notin([self.table.a, self.table.b, self.table.c]),
             '2 NOT IN (a, b, c)')
        ]
        self._check_expr_cases(cases)

    def test_isin_notin_in_select(self):
        filtered = self.table[self.table.g.isin(["foo", "bar"])]
        result = to_sql(filtered)
        expected = """SELECT *
FROM alltypes
WHERE g IN ('foo', 'bar')"""
        assert result == expected

        filtered = self.table[self.table.g.notin(["foo", "bar"])]
        result = to_sql(filtered)
        expected = """SELECT *
FROM alltypes
WHERE g NOT IN ('foo', 'bar')"""
        assert result == expected


class TestCoalesceGreaterLeast(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_coalesce(self):
        t = self.table
        cases = [
            (ibis.coalesce(t.string_col, 'foo'),
             "coalesce(string_col, 'foo')"),
            (ibis.coalesce(t.int_col, t.bigint_col),
             'coalesce(int_col, bigint_col)'),
        ]
        self._check_expr_cases(cases)

    def test_greatest(self):
        t = self.table
        cases = [
            (ibis.greatest(t.string_col, 'foo'),
             "greatest(string_col, 'foo')"),
            (ibis.greatest(t.int_col, t.bigint_col),
             'greatest(int_col, bigint_col)'),
        ]
        self._check_expr_cases(cases)

    def test_least(self):
        t = self.table
        cases = [
            (ibis.least(t.string_col, 'foo'),
             "least(string_col, 'foo')"),
            (ibis.least(t.int_col, t.bigint_col),
             'least(int_col, bigint_col)'),
        ]
        self._check_expr_cases(cases)


class TestStringBuiltins(unittest.TestCase, ExprSQLTest):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_unary_ops(self):
        cases = [
            (self.table.string_col.lower(), 'lower(string_col)'),
            (self.table.string_col.upper(), 'upper(string_col)'),
            (self.table.string_col.reverse(), 'reverse(string_col)'),
            (self.table.string_col.strip(), 'trim(string_col)'),
            (self.table.string_col.lstrip(), 'ltrim(string_col)'),
            (self.table.string_col.rstrip(), 'rtrim(string_col)'),
            (self.table.string_col.length(), 'length(string_col)'),
            (self.table.string_col.ascii_str(), 'ascii(string_col)')
        ]
        self._check_expr_cases(cases)

    def test_substr(self):
        # Database numbers starting from 1
        cases = [
            (self.table.string_col.substr(2), 'substr(string_col, 2 + 1)'),
            (self.table.string_col.substr(0, 3),
             'substr(string_col, 0 + 1, 3)')
        ]
        self._check_expr_cases(cases)

    def test_strright(self):
        cases = [
            (self.table.string_col.right(4), 'strright(string_col, 4)')
        ]
        self._check_expr_cases(cases)

    def test_like(self):
        cases = [
            (self.table.string_col.like('foo%'), "string_col LIKE 'foo%'")
        ]
        self._check_expr_cases(cases)

    def test_rlike(self):
        ex = "string_col RLIKE '[\d]+'"
        cases = [
            (self.table.string_col.rlike('[\d]+'), ex),
            (self.table.string_col.re_search('[\d]+'), ex),
        ]
        self._check_expr_cases(cases)

    def test_re_extract(self):
        sql = "regexp_extract(string_col, '[\d]+', 0)"
        cases = [
            (self.table.string_col.re_extract('[\d]+', 0), sql)
        ]
        self._check_expr_cases(cases)

    def test_re_replace(self):
        sql = "regexp_replace(string_col, '[\d]+', 'aaa')"
        cases = [
            (self.table.string_col.re_replace('[\d]+', 'aaa'), sql)
        ]
        self._check_expr_cases(cases)

    def test_repeat(self):
        cases = [
            (self.table.string_col.repeat(2), 'repeat(string_col, 2)')
        ]
        self._check_expr_cases(cases)

    def test_string_find(self):
        cases = [
            (self.table.string_col.find('a'), "instr(string_col, 'a') - 1")
        ]
        self._check_expr_cases(cases)

    def test_translate(self):
        cases = [
            (self.table.string_col.translate('a', 'b'),
             "translate(string_col, 'a', 'b')")
        ]
        self._check_expr_cases(cases)

    def test_locate(self):
        s = self.table.string_col
        i1 = self.table.tinyint_col
        cases = [
            (s.locate('a'), "locate('a', string_col) - 1"),
            (s.locate('a', 2), "locate('a', string_col, 3) - 1"),
            (s.locate('a', i1), "locate('a', string_col, tinyint_col + 1) - 1")
        ]
        self._check_expr_cases(cases)

    def test_lpad(self):
        cases = [
            (self.table.string_col.lpad(1, 'a'), "lpad(string_col, 1, 'a')"),
            (self.table.string_col.lpad(25, 'a'), "lpad(string_col, 25, 'a')")
        ]
        self._check_expr_cases(cases)

    def test_rpad(self):
        cases = [
            (self.table.string_col.rpad(1, 'a'), "rpad(string_col, 1, 'a')"),
            (self.table.string_col.rpad(25, 'a'), "rpad(string_col, 25, 'a')")
        ]
        self._check_expr_cases(cases)

    def test_find_in_set(self):
        cases = [
            (self.table.string_col.find_in_set(['a']),
             "find_in_set(string_col, 'a') - 1"),
            (self.table.string_col.find_in_set(['a', 'b']),
             "find_in_set(string_col, 'a,b') - 1")
        ]
        self._check_expr_cases(cases)

    def test_string_join(self):
        cases = [
            (ibis.literal(',').join(['a', 'b']), "concat_ws(',', 'a', 'b')")
        ]
        self._check_expr_cases(cases)
