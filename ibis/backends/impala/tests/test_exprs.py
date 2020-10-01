import unittest
from decimal import Decimal
from io import StringIO

import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis
import ibis.expr.api as api
import ibis.expr.types as ir
from ibis import literal as L
from ibis.common.exceptions import RelationError
from ibis.expr.datatypes import Category
from ibis.tests.expr.mocks import MockConnection
from ibis.tests.sql.test_compiler import ExprTestCases  # noqa: E402

from ..compiler import (  # noqa: E402
    ImpalaDialect,
    ImpalaExprTranslator,
    to_sql,
)

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')


pytestmark = pytest.mark.impala


def approx_equal(a, b, eps):
    assert abs(a - b) < eps


class ExprSQLTest:
    def _check_expr_cases(self, cases, named=False):
        for expr, expected in cases:
            repr(expr)
            result = self._translate(expr, named=named)
            assert result == expected

    def _translate(self, expr, context=None, named=False):
        if context is None:
            context = ImpalaDialect.make_context()
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
            ('An "escape"', "'An \"escape\"'"),
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
        cases = [(5, '5'), (1.5, '1.5'), (True, 'TRUE'), (False, 'FALSE')]
        self._check_literals(cases)

    def test_column_ref_table_aliases(self):
        context = ImpalaDialect.make_context()

        table1 = ibis.table([('key1', 'string'), ('value1', 'double')])

        table2 = ibis.table([('key2', 'string'), ('value and2', 'double')])

        context.set_ref(table1, 't0')
        context.set_ref(table2, 't1')

        expr = table1['value1'] - table2['value and2']

        result = self._translate(expr, context=context)
        expected = 't0.`value1` - t1.`value and2`'
        assert result == expected

    def test_column_ref_quoting(self):
        schema = [('has a space', 'double')]
        table = ibis.table(schema)
        self._translate(table['has a space'], named='`has a space`')

    def test_identifier_quoting(self):
        schema = [('date', 'double'), ('table', 'string')]
        table = ibis.table(schema)
        self._translate(table['date'], named='`date`')
        self._translate(table['table'], named='`table`')

    def test_named_expressions(self):
        a, b, g = self.table.get_columns(['a', 'b', 'g'])

        cases = [
            (g.cast('double').name('g_dub'), 'CAST(`g` AS double) AS `g_dub`'),
            (g.name('has a space'), '`g` AS `has a space`'),
            (((a - b) * a).name('expr'), '(`a` - `b`) * `a` AS `expr`'),
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
            (h ^ bool_col, '(`h` OR (`a` > 0)) AND NOT (`h` AND (`a` > 0))'),
        ]
        self._check_expr_cases(cases)

    def test_binary_infix_parenthesization(self):
        a, b, c = self.table.get_columns(['a', 'b', 'c'])

        cases = [
            ((a + b) + c, '(`a` + `b`) + `c`'),
            (a.log() + c, 'ln(`a`) + `c`'),
            (b + (-(a + c)), '`b` + (-(`a` + `c`))'),
        ]

        self._check_expr_cases(cases)

    def test_between(self):
        cases = [(self.table.f.between(0, 1), '`f` BETWEEN 0 AND 1')]
        self._check_expr_cases(cases)

    def test_isnull_notnull(self):
        cases = [
            (self.table['g'].isnull(), '`g` IS NULL'),
            (self.table['a'].notnull(), '`a` IS NOT NULL'),
            (
                (self.table['a'] + self.table['b']).isnull(),
                '`a` + `b` IS NULL',
            ),
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
            (g.cast('timestamp'), 'CAST(`g` AS timestamp)'),
        ]
        self._check_expr_cases(cases)

    def test_misc_conditionals(self):
        a = self.table.a
        cases = [(a.nullif(0), 'nullif(`a`, 0)')]
        self._check_expr_cases(cases)

    def test_decimal_casts(self):
        cases = [
            (
                L('9.9999999').cast('decimal(38, 5)'),
                "CAST('9.9999999' AS decimal(38, 5))",
            ),
            (
                self.table.f.cast('decimal(12, 2)'),
                "CAST(`f` AS decimal(12, 2))",
            ),
        ]
        self._check_expr_cases(cases)

    def test_negate(self):
        cases = [
            (-(self.table['a']), '-`a`'),
            (-(self.table['f']), '-`f`'),
            (-(self.table['h']), 'NOT `h`'),
        ]
        self._check_expr_cases(cases)

    def test_timestamp_extract_field(self):
        fields = [
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'millisecond',
        ]

        cases = [
            (
                getattr(self.table.i, field)(),
                "extract(`i`, '{0}')".format(field),
            )
            for field in fields
        ]
        self._check_expr_cases(cases)

        # integration with SQL translation
        expr = self.table[
            self.table.i.year().name('year'),
            self.table.i.month().name('month'),
            self.table.i.day().name('day'),
        ]

        result = to_sql(expr)
        expected = """SELECT extract(`i`, 'year') AS `year`, extract(`i`, 'month') AS `month`,
       extract(`i`, 'day') AS `day`
FROM alltypes"""
        assert result == expected

    def test_timestamp_now(self):
        cases = [(ibis.now(), 'now()')]
        self._check_expr_cases(cases)

    def test_timestamp_deltas(self):
        units = [
            ('years', 'year'),
            ('months', 'month'),
            ('weeks', 'week'),
            ('days', 'day'),
            ('hours', 'hour'),
            ('minutes', 'minute'),
            ('seconds', 'second'),
        ]

        t = self.table.i
        f = '`i`'

        cases = []
        for unit, compiled_unit in units:
            K = 5
            offset = ibis.interval(**{unit: K})
            add_template = 'date_add({1}, INTERVAL {2} {0})'
            sub_template = 'date_sub({1}, INTERVAL {2} {0})'

            cases.append(
                (t + offset, add_template.format(compiled_unit.upper(), f, K))
            )
            cases.append(
                (t - offset, sub_template.format(compiled_unit.upper(), f, K))
            )

        self._check_expr_cases(cases)

    def test_timestamp_literals(self):
        from pandas import Timestamp

        tv1 = '2015-01-01 12:34:56'
        ex1 = "'2015-01-01 12:34:56'"

        cases = [
            (L(Timestamp(tv1)), ex1),
            (L(Timestamp(tv1).to_pydatetime()), ex1),
            (ibis.timestamp(tv1), ex1),
        ]
        self._check_expr_cases(cases)

    def test_timestamp_day_of_week(self):
        timestamp_value = L('2015-09-01 01:00:23', type='timestamp')
        cases = [
            (
                timestamp_value.day_of_week.index(),
                "pmod(dayofweek('2015-09-01 01:00:23') - 2, 7)",
            ),
            (
                timestamp_value.day_of_week.full_name(),
                "dayname('2015-09-01 01:00:23')",
            ),
        ]
        self._check_expr_cases(cases)

    def test_timestamp_from_integer(self):
        col = self.table.c

        cases = [
            (
                col.to_timestamp(),
                'CAST(from_unixtime(`c`, "yyyy-MM-dd HH:mm:ss") '
                'AS timestamp)',
            ),
            (
                col.to_timestamp('ms'),
                'CAST(from_unixtime(CAST(floor(`c` / 1000) AS int), '
                '"yyyy-MM-dd HH:mm:ss") '
                'AS timestamp)',
            ),
            (
                col.to_timestamp('us'),
                'CAST(from_unixtime(CAST(floor(`c` / 1000000) AS int), '
                '"yyyy-MM-dd HH:mm:ss") '
                'AS timestamp)',
            ),
        ]
        self._check_expr_cases(cases)

    def test_correlated_predicate_subquery(self):
        t0 = self.table
        t1 = t0.view()

        expr = t0.g == t1.g

        ctx = ImpalaDialect.make_context()
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
            (bool_expr.any(), 'max(`f` = 0)'),
            (-(bool_expr.any()), 'max(`f` = 0) = FALSE'),
            (bool_expr.all(), 'min(`f` = 0)'),
            (-(bool_expr.all()), 'min(`f` = 0) = FALSE'),
        ]
        self._check_expr_cases(cases)


class TestUnaryBuiltins(unittest.TestCase, ExprSQLTest):
    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_numeric_unary_builtins(self):
        # No argument functions
        functions = [
            'abs',
            'ceil',
            'floor',
            'exp',
            'sqrt',
            ('log', 'ln'),
            ('approx_median', 'appx_median'),
            ('approx_nunique', 'ndv'),
            'ln',
            'log2',
            'log10',
            'nullifzero',
            'zeroifnull',
        ]

        cases = []
        for what in functions:
            if isinstance(what, tuple):
                ibis_name, sql_name = what
            else:
                ibis_name = sql_name = what

            for cname in ['double_col', 'int_col']:
                expr = getattr(self.table[cname], ibis_name)()
                cases.append(
                    (expr, '{0}({1})'.format(sql_name, '`{0}`'.format(cname)))
                )

        self._check_expr_cases(cases)

    def test_log_other_bases(self):
        cases = [(self.table.double_col.log(5), 'log(5, `double_col`)')]
        self._check_expr_cases(cases)

    def test_round(self):
        cases = [
            (self.table.double_col.round(), 'round(`double_col`)'),
            (self.table.double_col.round(0), 'round(`double_col`, 0)'),
            (self.table.double_col.round(2), 'round(`double_col`, 2)'),
            (
                self.table.double_col.round(self.table.tinyint_col),
                'round(`double_col`, `tinyint_col`)',
            ),
        ]
        self._check_expr_cases(cases)

    def test_sign(self):
        cases = [
            (
                self.table.tinyint_col.sign(),
                'CAST(sign(`tinyint_col`) AS tinyint)',
            ),
            (self.table.float_col.sign(), 'sign(`float_col`)'),
            (
                self.table.double_col.sign(),
                'CAST(sign(`double_col`) AS double)',
            ),
        ]
        self._check_expr_cases(cases)

    def test_hash(self):
        expr = self.table.int_col.hash()
        assert isinstance(expr, ir.IntegerColumn)
        assert isinstance(self.table.int_col.sum().hash(), ir.IntegerScalar)

        cases = [(self.table.int_col.hash(), 'fnv_hash(`int_col`)')]
        self._check_expr_cases(cases)

    def test_reduction_where(self):
        cond = self.table.bigint_col < 70
        c = self.table.double_col
        tmp = (
            '{0}(CASE WHEN `bigint_col` < 70 THEN `double_col` '
            'ELSE NULL END)'
        )
        cases = [
            (c.sum(where=cond), tmp.format('sum')),
            (c.count(where=cond), tmp.format('count')),
            (c.mean(where=cond), tmp.format('avg')),
            (c.max(where=cond), tmp.format('max')),
            (c.min(where=cond), tmp.format('min')),
            (c.std(where=cond), tmp.format('stddev_samp')),
            (c.std(where=cond, how='pop'), tmp.format('stddev_pop')),
            (c.var(where=cond), tmp.format('var_samp')),
            (c.var(where=cond, how='pop'), tmp.format('var_pop')),
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
  ELSE CAST(NULL AS bigint)
END"""
        assert result == expected

    def test_where_use_if(self):
        expr = ibis.where(self.table.f > 0, self.table.e, self.table.a)
        assert isinstance(expr, ir.FloatingValue)

        result = self._translate(expr)
        expected = "if(`f` > 0, `e`, `a`)"
        assert result == expected

    def test_nullif_ifnull(self):
        table = self.con.table('tpch_lineitem')

        f = table.l_quantity

        cases = [
            (f.nullif(f), 'nullif(`l_quantity`, `l_quantity`)'),
            (
                (f == 0).nullif(f == 0),
                'nullif(`l_quantity` = 0, `l_quantity` = 0)',
            ),
            (
                (f != 0).nullif(f == 0),
                'nullif(`l_quantity` != 0, `l_quantity` = 0)',
            ),
            (f.fillna(0), 'isnull(`l_quantity`, CAST(0 AS decimal(12, 2)))'),
        ]
        self._check_expr_cases(cases)

    def test_decimal_fillna_cast_arg(self):
        table = self.con.table('tpch_lineitem')
        f = table.l_extendedprice

        cases = [
            (
                f.fillna(0),
                'isnull(`l_extendedprice`, CAST(0 AS decimal(12, 2)))',
            ),
            (f.fillna(0.0), 'isnull(`l_extendedprice`, 0.0)'),
        ]
        self._check_expr_cases(cases)

    def test_identical_to(self):
        t = self.con.table('functional_alltypes')
        expr = t.tinyint_col.identical_to(t.double_col)
        result = to_sql(expr)
        expected = """\
SELECT `tinyint_col` IS NOT DISTINCT FROM `double_col` AS `tmp`
FROM functional_alltypes"""
        assert result == expected

    def test_identical_to_special_case(self):
        expr = ibis.NA.cast('int64').identical_to(ibis.NA.cast('int64'))
        result = to_sql(expr)
        assert result == 'SELECT TRUE AS `tmp`'


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
  ELSE CAST(NULL AS tinyint)
END"""

        expr2 = self.table.f.bucket(buckets, close_extreme=False)
        expected2 = """\
CASE
  WHEN (`f` >= 0) AND (`f` < 10) THEN 0
  WHEN (`f` >= 10) AND (`f` < 25) THEN 1
  WHEN (`f` >= 25) AND (`f` < 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END"""

        expr3 = self.table.f.bucket(buckets, closed='right')
        expected3 = """\
CASE
  WHEN (`f` >= 0) AND (`f` <= 10) THEN 0
  WHEN (`f` > 10) AND (`f` <= 25) THEN 1
  WHEN (`f` > 25) AND (`f` <= 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END"""

        expr4 = self.table.f.bucket(
            buckets, closed='right', close_extreme=False
        )
        expected4 = """\
CASE
  WHEN (`f` > 0) AND (`f` <= 10) THEN 0
  WHEN (`f` > 10) AND (`f` <= 25) THEN 1
  WHEN (`f` > 25) AND (`f` <= 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END"""

        expr5 = self.table.f.bucket(buckets, include_under=True)
        expected5 = """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (`f` >= 0) AND (`f` < 10) THEN 1
  WHEN (`f` >= 10) AND (`f` < 25) THEN 2
  WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
  ELSE CAST(NULL AS tinyint)
END"""

        expr6 = self.table.f.bucket(
            buckets, include_under=True, include_over=True
        )
        expected6 = """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (`f` >= 0) AND (`f` < 10) THEN 1
  WHEN (`f` >= 10) AND (`f` < 25) THEN 2
  WHEN (`f` >= 25) AND (`f` <= 50) THEN 3
  WHEN `f` > 50 THEN 4
  ELSE CAST(NULL AS tinyint)
END"""

        expr7 = self.table.f.bucket(
            buckets, close_extreme=False, include_under=True, include_over=True
        )
        expected7 = """\
CASE
  WHEN `f` < 0 THEN 0
  WHEN (`f` >= 0) AND (`f` < 10) THEN 1
  WHEN (`f` >= 10) AND (`f` < 25) THEN 2
  WHEN (`f` >= 25) AND (`f` < 50) THEN 3
  WHEN `f` >= 50 THEN 4
  ELSE CAST(NULL AS tinyint)
END"""

        expr8 = self.table.f.bucket(
            buckets, closed='right', close_extreme=False, include_under=True
        )
        expected8 = """\
CASE
  WHEN `f` <= 0 THEN 0
  WHEN (`f` > 0) AND (`f` <= 10) THEN 1
  WHEN (`f` > 10) AND (`f` <= 25) THEN 2
  WHEN (`f` > 25) AND (`f` <= 50) THEN 3
  ELSE CAST(NULL AS tinyint)
END"""

        expr9 = self.table.f.bucket(
            [10], closed='right', include_over=True, include_under=True
        )
        expected9 = """\
CASE
  WHEN `f` <= 10 THEN 0
  WHEN `f` > 10 THEN 1
  ELSE CAST(NULL AS tinyint)
END"""

        expr10 = self.table.f.bucket(
            [10], include_over=True, include_under=True
        )
        expected10 = """\
CASE
  WHEN `f` < 10 THEN 0
  WHEN `f` >= 10 THEN 1
  ELSE CAST(NULL AS tinyint)
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
        expr = self.table.f.bucket(
            [10], include_over=True, include_under=True
        ).cast('int32')

        expected = """\
CASE
  WHEN `f` < 10 THEN 0
  WHEN `f` >= 10 THEN 1
  ELSE CAST(NULL AS tinyint)
END"""

        expr2 = self.table.f.bucket(
            [10], include_over=True, include_under=True
        ).cast('double')

        expected2 = """\
CAST(CASE
  WHEN `f` < 10 THEN 0
  WHEN `f` >= 10 THEN 1
  ELSE CAST(NULL AS tinyint)
END AS double)"""

        self._check_expr_cases([(expr, expected), (expr2, expected2)])

    def test_bucket_assign_labels(self):
        buckets = [0, 10, 25, 50]
        bucket = self.table.f.bucket(buckets, include_under=True)

        size = self.table.group_by(bucket.name('tier')).size()
        labelled = size.tier.label(
            ['Under 0', '0 to 10', '10 to 25', '25 to 50'], nulls='error'
        ).name('tier2')
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
      ELSE CAST(NULL AS tinyint)
    END AS `tier`, count(*) AS `count`
  FROM alltypes
  GROUP BY 1
) t0"""

        result = to_sql(expr)

        assert result == expected

        self.assertRaises(ValueError, size.tier.label, ['a', 'b', 'c'])
        self.assertRaises(
            ValueError, size.tier.label, ['a', 'b', 'c', 'd', 'e']
        )


class TestInNotIn(unittest.TestCase, ExprSQLTest):
    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_field_in_literals(self):
        values = ['foo', 'bar', 'baz']
        values_formatted = tuple(set(values))
        cases = [
            (self.table.g.isin(values), "`g` IN {}".format(values_formatted)),
            (
                self.table.g.notin(values),
                "`g` NOT IN {}".format(values_formatted),
            ),
        ]
        self._check_expr_cases(cases)

    def test_literal_in_list(self):
        cases = [
            (
                L(2).isin([self.table.a, self.table.b, self.table.c]),
                '2 IN (`a`, `b`, `c`)',
            ),
            (
                L(2).notin([self.table.a, self.table.b, self.table.c]),
                '2 NOT IN (`a`, `b`, `c`)',
            ),
        ]
        self._check_expr_cases(cases)

    def test_isin_notin_in_select(self):
        values = ['foo', 'bar']
        values_formatted = tuple(set(values))

        filtered = self.table[self.table.g.isin(values)]
        result = to_sql(filtered)
        expected = """SELECT *
FROM alltypes
WHERE `g` IN {}"""
        assert result == expected.format(values_formatted)

        filtered = self.table[self.table.g.notin(values)]
        result = to_sql(filtered)
        expected = """SELECT *
FROM alltypes
WHERE `g` NOT IN {}"""
        assert result == expected.format(values_formatted)


class TestCoalesceGreaterLeast(unittest.TestCase, ExprSQLTest):
    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('functional_alltypes')

    def test_coalesce(self):
        t = self.table
        cases = [
            (
                ibis.coalesce(t.string_col, 'foo'),
                "coalesce(`string_col`, 'foo')",
            ),
            (
                ibis.coalesce(t.int_col, t.bigint_col),
                'coalesce(`int_col`, `bigint_col`)',
            ),
        ]
        self._check_expr_cases(cases)

    def test_greatest(self):
        t = self.table
        cases = [
            (
                ibis.greatest(t.string_col, 'foo'),
                "greatest(`string_col`, 'foo')",
            ),
            (
                ibis.greatest(t.int_col, t.bigint_col),
                'greatest(`int_col`, `bigint_col`)',
            ),
        ]
        self._check_expr_cases(cases)

    def test_least(self):
        t = self.table
        cases = [
            (ibis.least(t.string_col, 'foo'), "least(`string_col`, 'foo')"),
            (
                ibis.least(t.int_col, t.bigint_col),
                'least(`int_col`, `bigint_col`)',
            ),
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
            (
                ibis.row_number().over(w),
                '(row_number() OVER (ORDER BY `float_col`) - 1)',
            ),
            (t.string_col.lag(), 'lag(`string_col`)'),
            (t.string_col.lag(2), 'lag(`string_col`, 2)'),
            (t.string_col.lag(default=0), 'lag(`string_col`, 1, 0)'),
            (t.string_col.lead(), 'lead(`string_col`)'),
            (t.string_col.lead(2), 'lead(`string_col`, 2)'),
            (t.string_col.lead(default=0), 'lead(`string_col`, 1, 0)'),
            (t.double_col.first(), 'first_value(`double_col`)'),
            (t.double_col.last(), 'last_value(`double_col`)'),
            # (t.double_col.nth(4), 'first_value(lag(double_col, 4 - 1))')
            (t.double_col.ntile(3), 'ntile(3)'),
            (t.double_col.percent_rank(), 'percent_rank()'),
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
            (s.ascii_str(), 'ascii(`string_col`)'),
        ]
        self._check_expr_cases(cases)

    def test_substr(self):
        # Database numbers starting from 1
        cases = [
            (self.table.string_col.substr(2), 'substr(`string_col`, 2 + 1)'),
            (
                self.table.string_col.substr(0, 3),
                'substr(`string_col`, 0 + 1, 3)',
            ),
        ]
        self._check_expr_cases(cases)

    def test_strright(self):
        cases = [(self.table.string_col.right(4), 'strright(`string_col`, 4)')]
        self._check_expr_cases(cases)

    def test_like(self):
        cases = [
            (self.table.string_col.like('foo%'), "`string_col` LIKE 'foo%'"),
            (
                self.table.string_col.like(['foo%', '%bar']),
                "`string_col` LIKE 'foo%' OR `string_col` LIKE '%bar'",
            ),
        ]
        self._check_expr_cases(cases)

    def test_rlike(self):
        ex = r"regexp_like(`string_col`, '[\d]+')"
        cases = [
            (self.table.string_col.rlike(r'[\d]+'), ex),
            (self.table.string_col.re_search(r'[\d]+'), ex),
        ]
        self._check_expr_cases(cases)

    def test_re_extract(self):
        sql = r"regexp_extract(`string_col`, '[\d]+', 0)"
        cases = [(self.table.string_col.re_extract(r'[\d]+', 0), sql)]
        self._check_expr_cases(cases)

    def test_re_replace(self):
        sql = r"regexp_replace(`string_col`, '[\d]+', 'aaa')"
        cases = [(self.table.string_col.re_replace(r'[\d]+', 'aaa'), sql)]
        self._check_expr_cases(cases)

    def test_parse_url(self):
        sql = "parse_url(`string_col`, 'HOST')"
        cases = [(self.table.string_col.parse_url('HOST'), sql)]
        self._check_expr_cases(cases)

    def test_repeat(self):
        cases = [(self.table.string_col.repeat(2), 'repeat(`string_col`, 2)')]
        self._check_expr_cases(cases)

    def test_translate(self):
        cases = [
            (
                self.table.string_col.translate('a', 'b'),
                "translate(`string_col`, 'a', 'b')",
            )
        ]
        self._check_expr_cases(cases)

    def test_find(self):
        s = self.table.string_col
        i1 = self.table.tinyint_col
        cases = [
            (s.find('a'), "locate('a', `string_col`) - 1"),
            (s.find('a', 2), "locate('a', `string_col`, 3) - 1"),
            (
                s.find('a', start=i1),
                "locate('a', `string_col`, `tinyint_col` + 1) - 1",
            ),
        ]
        self._check_expr_cases(cases)

    def test_lpad(self):
        cases = [
            (self.table.string_col.lpad(1, 'a'), "lpad(`string_col`, 1, 'a')"),
            (self.table.string_col.lpad(25), "lpad(`string_col`, 25, ' ')"),
        ]
        self._check_expr_cases(cases)

    def test_rpad(self):
        cases = [
            (self.table.string_col.rpad(1, 'a'), "rpad(`string_col`, 1, 'a')"),
            (self.table.string_col.rpad(25), "rpad(`string_col`, 25, ' ')"),
        ]
        self._check_expr_cases(cases)

    def test_find_in_set(self):
        cases = [
            (
                self.table.string_col.find_in_set(['a']),
                "find_in_set(`string_col`, 'a') - 1",
            ),
            (
                self.table.string_col.find_in_set(['a', 'b']),
                "find_in_set(`string_col`, 'a,b') - 1",
            ),
        ]
        self._check_expr_cases(cases)

    def test_string_join(self):
        cases = [(L(',').join(['a', 'b']), "concat_ws(',', 'a', 'b')")]
        self._check_expr_cases(cases)


def test_embedded_identifier_quoting(alltypes):
    t = alltypes

    expr = t[[(t.double_col * 2).name('double(fun)')]]['double(fun)'].sum()
    expr.execute()


def test_table_info(alltypes):
    buf = StringIO()
    alltypes.info(buf=buf)

    assert buf.getvalue() is not None


@pytest.mark.parametrize(('expr', 'expected'), [(L(1) + L(2), 3)])
def test_execute_exprs_no_table_ref(con, expr, expected):
    result = con.execute(expr)
    assert result == expected

    # ExprList
    exlist = ibis.api.expr_list(
        [L(1).name('a'), ibis.now().name('b'), L(2).log().name('c')]
    )
    con.execute(exlist)


def test_summary_execute(alltypes):
    table = alltypes

    # also test set_column while we're at it
    table = table.set_column('double_col', table.double_col * 2)

    expr = table.double_col.summary()
    repr(expr)

    result = expr.execute()
    assert isinstance(result, pd.DataFrame)

    expr = table.group_by('string_col').aggregate(
        [
            table.double_col.summary().prefix('double_'),
            table.float_col.summary().prefix('float_'),
            table.string_col.summary().suffix('_string'),
        ]
    )
    result = expr.execute()
    assert isinstance(result, pd.DataFrame)


def test_distinct_array(con, alltypes):
    table = alltypes

    expr = table.string_col.distinct()
    result = con.execute(expr)
    assert isinstance(result, pd.Series)


def test_decimal_metadata(con):
    table = con.table('tpch_lineitem')

    expr = table.l_quantity
    assert expr.type().precision == 12
    assert expr.type().scale == 2

    # TODO: what if user impyla version does not have decimal Metadata?


def test_builtins_1(con, alltypes):
    table = alltypes

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
        api.coalesce(
            table.int_col, api.null(), table.smallint_col, table.bigint_col, 5
        ),
        api.greatest(table.float_col, table.double_col, 5),
        api.least(table.string_col, 'foo'),
        # string stuff
        s.contains('6'),
        s.like('6%'),
        s.re_search(r'[\d]+'),
        s.re_extract(r'[\d]+', 0),
        s.re_replace(r'[\d]+', 'a'),
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
        s.repeat(i1),
    ]

    proj_exprs = [expr.name('e%d' % i) for i, expr in enumerate(exprs)]

    projection = table[proj_exprs]
    projection.limit(10).execute()

    _check_impala_output_types_match(con, projection)


def _check_impala_output_types_match(con, table):
    query = to_sql(table)
    t = con.sql(query)

    def _clean_type(x):
        if isinstance(x, Category):
            x = x.to_integer_type()
        return x

    left, right = t.schema(), table.schema()
    for i, (n, left, right) in enumerate(
        zip(left.names, left.types, right.types)
    ):
        left = _clean_type(left)
        right = _clean_type(right)

        if left != right:
            pytest.fail(
                'Value for {0} had left type {1}'
                ' and right type {2}'.format(n, left, right)
            )


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        # mod cases
        (L(50) % 5, 0),
        (L(50000) % 10, 0),
        (250 % L(50), 0),
        # nullif cases
        (5 / L(50).nullif(0), 0.1),
        (5 / L(50).nullif(L(50000)), 0.1),
        (5 / L(50000).nullif(0), 0.0001),
        (L(50000).zeroifnull(), 50000),
    ],
)
def test_int_builtins(con, expr, expected):
    result = con.execute(expr)
    assert result == expected, to_sql(expr)


def test_column_types(alltypes):
    df = alltypes.execute()
    assert df.tinyint_col.dtype.name == 'int8'
    assert df.smallint_col.dtype.name == 'int16'
    assert df.int_col.dtype.name == 'int32'
    assert df.bigint_col.dtype.name == 'int64'
    assert df.float_col.dtype.name == 'float32'
    assert df.double_col.dtype.name == 'float64'
    assert df.timestamp_col.dtype.name == 'datetime64[ns]'


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L(50000).to_timestamp('s'), pd.to_datetime(50000, unit='s')),
        (L(50000).to_timestamp('ms'), pd.to_datetime(50000, unit='ms')),
        (L(5 * 10 ** 8).to_timestamp(), pd.to_datetime(5 * 10 ** 8, unit='s')),
        (
            ibis.timestamp('2009-05-17 12:34:56').truncate('y'),
            pd.Timestamp('2009-01-01'),
        ),
        (
            ibis.timestamp('2009-05-17 12:34:56').truncate('M'),
            pd.Timestamp('2009-05-01'),
        ),
        (
            ibis.timestamp('2009-05-17 12:34:56').truncate('month'),
            pd.Timestamp('2009-05-01'),
        ),
        (
            ibis.timestamp('2009-05-17 12:34:56').truncate('d'),
            pd.Timestamp('2009-05-17'),
        ),
        (
            ibis.timestamp('2009-05-17 12:34:56').truncate('h'),
            pd.Timestamp('2009-05-17 12:00'),
        ),
        (
            ibis.timestamp('2009-05-17 12:34:56').truncate('m'),
            pd.Timestamp('2009-05-17 12:34'),
        ),
        (
            ibis.timestamp('2009-05-17 12:34:56').truncate('minute'),
            pd.Timestamp('2009-05-17 12:34'),
        ),
    ],
)
def test_timestamp_builtins(con, expr, expected):
    result = con.execute(expr)
    assert result == expected, to_sql(expr)


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L(-5).abs(), 5),
        (L(5.245).cast('int32'), 5),
        (L(5.245).ceil(), 6),
        (L(5.245).isnull(), False),
        (L(5.245).floor(), 5),
        (L(5.245).notnull(), True),
        (L(5.245).round(), 5),
        (L(5.245).round(2), Decimal('5.25')),
        (L(5.245).sign(), 1),
    ],
)
def test_decimal_builtins(con, expr, expected):
    result = con.execute(expr)
    assert result == expected, to_sql(expr)


@pytest.mark.parametrize(
    ('func', 'expected'),
    [
        pytest.param(lambda dc: dc, '5.245', id='id'),
        pytest.param(lambda dc: dc % 5, '0.245', id='mod'),
        pytest.param(lambda dc: dc.fillna(0), '5.245', id='fillna'),
        pytest.param(lambda dc: dc.exp(), '189.6158', id='exp'),
        pytest.param(lambda dc: dc.log(), '1.65728', id='log'),
        pytest.param(lambda dc: dc.log2(), '2.39094', id='log2'),
        pytest.param(lambda dc: dc.log10(), '0.71975', id='log10'),
        pytest.param(lambda dc: dc.sqrt(), '2.29019', id='sqrt'),
        pytest.param(lambda dc: dc.zeroifnull(), '5.245', id='zeroifnull'),
        pytest.param(lambda dc: -dc, '-5.245', id='neg'),
    ],
)
def test_decimal_builtins_2(con, func, expected):
    dc = L('5.245').cast('decimal(12, 5)')
    expr = func(dc)
    result = con.execute(expr)
    tol = Decimal('0.0001')
    approx_equal(Decimal(result), Decimal(expected), tol)


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L('abcd').length(), 4),
        (L('ABCD').lower(), 'abcd'),
        (L('abcd').upper(), 'ABCD'),
        (L('abcd').reverse(), 'dcba'),
        (L('abcd').ascii_str(), 97),
        (L('   a   ').strip(), 'a'),
        (L('   a   ').lstrip(), 'a   '),
        (L('   a   ').rstrip(), '   a'),
        (L('abcd').capitalize(), 'Abcd'),
        (L('abcd').substr(0, 2), 'ab'),
        (L('abcd').left(2), 'ab'),
        (L('abcd').right(2), 'cd'),
        (L('abcd').repeat(2), 'abcdabcd'),
        # global replace not available in Impala yet
        # (L('aabbaabbaa').replace('bb', 'B'), 'aaBaaBaa'),
        (L('0123').translate('012', 'abc'), 'abc3'),
        (L('abcd').find('a'), 0),
        (L('baaaab').find('b', 2), 5),
        (L('abcd').lpad(1, '-'), 'a'),
        (L('abcd').lpad(5), ' abcd'),
        (L('abcd').rpad(1, '-'), 'a'),
        (L('abcd').rpad(5), 'abcd '),
        (L('abcd').find_in_set(['a', 'b', 'abcd']), 2),
        (L(', ').join(['a', 'b']), 'a, b'),
        (L('abcd').like('a%'), True),
        (L('abcd').re_search('[a-z]'), True),
        (L('abcd').re_extract('[a-z]', 0), 'a'),
        (L('abcd').re_replace('(b)', '2'), 'a2cd'),
    ],
)
def test_string_functions(con, expr, expected):
    result = con.execute(expr)
    assert result == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L("https://www.cloudera.com").parse_url('HOST'), "www.cloudera.com"),
        (
            L('https://www.youtube.com/watch?v=kEuEcWfewf8&t=10').parse_url(
                'QUERY', 'v'
            ),
            'kEuEcWfewf8',
        ),
    ],
)
def test_parse_url(con, expr, expected):
    result = con.execute(expr)
    assert result == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L(7) / 2, 3.5),
        (L(7) // 2, 3),
        (L(7).floordiv(2), 3),
        (L(2).rfloordiv(7), 3),
    ],
)
def test_div_floordiv(con, expr, expected):
    result = con.execute(expr)
    assert result == expected


@pytest.mark.xfail(
    raises=RelationError,
    reason='Equality was broken, and fixing it broke this test',
)
def test_filter_predicates(con):
    t = con.table('tpch_nation')

    predicates = [
        lambda x: x.n_name.lower().like('%ge%'),
        lambda x: x.n_name.lower().contains('ge'),
        lambda x: x.n_name.lower().rlike('.*ge.*'),
    ]

    expr = t
    for pred in predicates:
        expr = expr[pred(expr)].projection([expr])

    expr.execute()


def test_histogram_value_counts(alltypes):
    t = alltypes
    expr = t.double_col.histogram(10).value_counts()
    expr.execute()


def test_casted_expr_impala_bug(alltypes):
    # Per GH #396. Prior to Impala 2.3.0, there was a bug in the query
    # planner that caused this expression to fail
    expr = alltypes.string_col.cast('double').value_counts()
    expr.execute()


def test_decimal_timestamp_builtins(con):
    table = con.table('tpch_lineitem')

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
        api.where(table.l_discount > 0, dc * table.l_discount, api.NA),
        dc.fillna(0),
        ts < (ibis.now() + ibis.interval(months=3)),
        ts < (ibis.timestamp('2005-01-01') + ibis.interval(months=3)),
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

    timestamp_fields = [
        'years',
        'months',
        'days',
        'hours',
        'minutes',
        'seconds',
        'weeks',
    ]
    for field in timestamp_fields:
        if hasattr(ts, field):
            exprs.append(getattr(ts, field)())

        offset = ibis.interval(**{field: 2})
        exprs.append(ts + offset)
        exprs.append(ts - offset)

    proj_exprs = [expr.name('e%d' % i) for i, expr in enumerate(exprs)]

    projection = table[proj_exprs].limit(10)
    projection.execute()


def test_timestamp_scalar_in_filter(alltypes):
    # #310
    table = alltypes

    expr = table.filter(
        [
            table.timestamp_col
            < (ibis.timestamp('2010-01-01') + ibis.interval(months=3)),
            table.timestamp_col < (ibis.now() + ibis.interval(days=10)),
        ]
    ).count()
    expr.execute()


def test_aggregations(alltypes):
    table = alltypes.limit(100)

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

    metrics = [expr.name('e%d' % i) for i, expr in enumerate(exprs)]

    agged_table = table.aggregate(metrics)
    agged_table.execute()


def test_analytic_functions(alltypes):
    t = alltypes.limit(1000)

    g = t.group_by('string_col').order_by('double_col')
    f = t.float_col

    exprs = [
        f.lag(),
        f.lead(),
        f.rank(),
        f.dense_rank(),
        f.percent_rank(),
        f.ntile(buckets=7),
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
        f.max(),
    ]

    proj_exprs = [expr.name('e%d' % i) for i, expr in enumerate(exprs)]

    proj_table = g.mutate(proj_exprs)
    proj_table.execute()


def test_anti_join_self_reference_works(con, alltypes):
    t = alltypes.limit(100)
    t2 = t.view()
    case = t[-((t.string_col == t2.string_col).any())]
    con.explain(case)


def test_tpch_self_join_failure(con):
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    customer = con.table('tpch_customer')
    orders = con.table('tpch_orders')

    fields_of_interest = [
        region.r_name.name('region'),
        nation.n_name.name('nation'),
        orders.o_totalprice.name('amount'),
        orders.o_orderdate.cast('timestamp').name('odate'),
    ]

    joined_all = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)[
            fields_of_interest
        ]
    )

    year = joined_all.odate.year().name('year')
    total = joined_all.amount.sum().cast('double').name('total')
    annual_amounts = joined_all.group_by(['region', year]).aggregate(total)

    current = annual_amounts
    prior = annual_amounts.view()

    yoy_change = (current.total - prior.total).name('yoy_change')
    yoy = current.join(
        prior,
        (
            (current.region == prior.region)
            & (current.year == (prior.year - 1))
        ),
    )[current.region, current.year, yoy_change]

    # no analysis failure
    con.explain(yoy)


def test_tpch_correlated_subquery_failure(con):
    # #183 and other issues
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')
    customer = con.table('tpch_customer')
    orders = con.table('tpch_orders')

    fields_of_interest = [
        customer,
        region.r_name.name('region'),
        orders.o_totalprice.name('amount'),
        orders.o_orderdate.cast('timestamp').name('odate'),
    ]

    tpch = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)[
            fields_of_interest
        ]
    )

    t2 = tpch.view()
    conditional_avg = t2[(t2.region == tpch.region)].amount.mean()
    amount_filter = tpch.amount > conditional_avg

    expr = tpch[amount_filter].limit(0)
    con.explain(expr)


def test_non_equijoin(con):
    t = con.table('functional_alltypes').limit(100)
    t2 = t.view()

    expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()

    # it works
    expr.execute()


def test_char_varchar_types(con):
    sql = """\
SELECT CAST(string_col AS varchar(20)) AS varchar_col,
   CAST(string_col AS CHAR(5)) AS char_col
FROM functional_alltypes"""

    t = con.sql(sql)

    assert isinstance(t.varchar_col, ir.StringColumn)
    assert isinstance(t.char_col, ir.StringColumn)


def test_unions_with_ctes(con, alltypes):
    t = alltypes

    expr1 = t.group_by(['tinyint_col', 'string_col']).aggregate(
        t.double_col.sum().name('metric')
    )
    expr2 = expr1.view()

    join1 = expr1.join(expr2, expr1.string_col == expr2.string_col)[[expr1]]
    join2 = join1.view()

    expr = join1.union(join2)
    con.explain(expr)


def test_head(con):
    t = con.table('functional_alltypes')
    result = t.head().execute()
    expected = t.limit(5).execute()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (ibis.NA.cast('int64'), ibis.NA.cast('int64'), True),
        (L(1), L(1), True),
        (ibis.NA.cast('int64'), L(1), False),
        (L(1), ibis.NA.cast('int64'), False),
        (L(0), L(1), False),
        (L(1), L(0), False),
    ],
)
def test_identical_to(con, left, right, expected):
    expr = left.identical_to(right)
    result = con.execute(expr)
    assert result == expected


def test_not(alltypes):
    t = alltypes.limit(10)
    expr = t.projection([(~t.double_col.isnull()).name('double_col')])
    result = expr.execute().double_col
    expected = ~t.execute().double_col.isnull()
    tm.assert_series_equal(result, expected)


def test_where_with_timestamp():
    t = ibis.table(
        [('uuid', 'string'), ('ts', 'timestamp'), ('search_level', 'int64')],
        name='t',
    )
    expr = t.group_by(t.uuid).aggregate(
        min_date=t.ts.min(where=t.search_level == 1)
    )
    result = ibis.impala.compile(expr)
    expected = """\
SELECT `uuid`,
       min(CASE WHEN `search_level` = 1 THEN `ts` ELSE NULL END) AS `min_date`
FROM t
GROUP BY 1"""
    assert result == expected


def test_filter_with_analytic():
    x = ibis.table(ibis.schema([('col', 'int32')]), 'x')
    with_filter_col = x[x.columns + [ibis.null().name('filter')]]
    filtered = with_filter_col[with_filter_col['filter'].isnull()]
    subquery = filtered[filtered.columns]

    with_analytic = subquery[['col', subquery.count().name('analytic')]]
    expr = with_analytic[with_analytic.columns]

    result = ibis.impala.compile(expr)
    expected = """\
SELECT `col`, `analytic`
FROM (
  SELECT `col`, count(*) OVER () AS `analytic`
  FROM (
    SELECT `col`, `filter`
    FROM (
      SELECT *
      FROM (
        SELECT `col`, NULL AS `filter`
        FROM x
      ) t3
      WHERE `filter` IS NULL
    ) t2
  ) t1
) t0"""

    assert result == expected


def test_named_from_filter_groupby():
    t = ibis.table([('key', 'string'), ('value', 'double')], name='t0')
    gb = t.filter(t.value == 42).groupby(t.key)
    sum_expr = lambda t: (t.value + 1 + 2 + 3).sum()  # noqa: E731
    expr = gb.aggregate(abc=sum_expr)
    expected = """\
SELECT `key`, sum(((`value` + 1) + 2) + 3) AS `abc`
FROM t0
WHERE `value` = 42
GROUP BY 1"""
    assert ibis.impala.compile(expr) == expected

    expr = gb.aggregate(foo=sum_expr)
    expected = """\
SELECT `key`, sum(((`value` + 1) + 2) + 3) AS `foo`
FROM t0
WHERE `value` = 42
GROUP BY 1"""
    assert ibis.impala.compile(expr) == expected


def test_nunique_where():
    t = ibis.table([('key', 'string'), ('value', 'double')], name='t0')
    expr = t.key.nunique(where=t.value >= 1.0)
    expected = """\
SELECT count(DISTINCT CASE WHEN `value` >= 1.0 THEN `key` ELSE NULL END) AS `nunique`
FROM t0"""  # noqa: E501
    result = ibis.impala.compile(expr)
    assert result == expected
