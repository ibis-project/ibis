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

import unittest

from ibis.sql.exprs import ExprTranslator
from ibis.sql.compiler import QueryContext, to_sql
from ibis.expr.tests.mocks import MockConnection
import ibis as api
import ibis.expr.types as ir


class ExprSQLTest(object):

    def _check_expr_cases(self, cases, context=None, named=False):
        for expr, expected in cases:
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
            lit_expr = api.literal(value)
            result = self._translate(lit_expr)
            assert result == expected

    def test_string_literals(self):
        cases = [
            ('simple', "'simple'"),
            ('I can\'t', "'I can\\'t'"),
            ('An "escape"', "'An \"escape\"'")
        ]

        for value, expected in cases:
            lit_expr = api.literal(value)
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

        table1 = api.table([
            ('key1', 'string'),
            ('value1', 'double')
        ])

        table2 = api.table([
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
        table = api.table(schema)
        self._translate(table['has a space'], '`has a space`')

    def test_identifier_quoting(self):
        schema = [('date', 'double'), ('table', 'string')]
        table = api.table(schema)
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
            (api.literal('9.9999999').cast('decimal(38,5)'),
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
                  "extract(i, '{}')".format(field))
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
            (api.now(), 'now()')
        ]
        self._check_expr_cases(cases)

    def test_timestamp_deltas(self):
        units = ['year', 'month', 'week', 'day',
                 'hour', 'minute', 'second',
                 'millisecond', 'microsecond', 'nanosecond']

        t = self.table.i
        f = 'i'

        cases = []
        for unit in units:
            K = 5
            offset = getattr(api, unit)(K)
            template = '{}s_add({}, {})'

            cases.append((t + offset, template.format(unit, f, K)))
            cases.append((t - offset, template.format(unit, f, -K)))

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
                cases.append((expr, '{}({})'.format(sql_name, cname)))

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
            (self.table.double_col.round(2, ), 'round(double_col, 2)')
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
        expr = (api.case()
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
        expr = api.where(self.table.f > 0, self.table.e, self.table.a)
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
            (f.fillna(0), 'isnull(l_quantity, 0)'),
        ]
        self._check_expr_cases(cases)


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
            (api.literal(2).isin([self.table.a, self.table.b, self.table.c]),
             '2 IN (a, b, c)'),
            (api.literal(2).notin([self.table.a, self.table.b, self.table.c]),
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
            (api.coalesce(t.string_col, 'foo'),
             "coalesce(string_col, 'foo')"),
            (api.coalesce(t.int_col, t.bigint_col),
             'coalesce(int_col, bigint_col)'),
        ]
        self._check_expr_cases(cases)

    def test_greatest(self):
        t = self.table
        cases = [
            (api.greatest(t.string_col, 'foo'),
             "greatest(string_col, 'foo')"),
            (api.greatest(t.int_col, t.bigint_col),
             'greatest(int_col, bigint_col)'),
        ]
        self._check_expr_cases(cases)

    def test_least(self):
        t = self.table
        cases = [
            (api.least(t.string_col, 'foo'),
             "least(string_col, 'foo')"),
            (api.least(t.int_col, t.bigint_col),
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
            (self.table.string_col.length(), 'length(string_col)')
        ]
        self._check_expr_cases(cases)

    def test_substr(self):
        # Database numbers starting from 1
        cases = [
            (self.table.string_col.substr(2), 'substr(string_col, 3)'),
            (self.table.string_col.substr(0, 3), 'substr(string_col, 1, 3)')
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
