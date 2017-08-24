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
import sys
import math
import operator

from operator import methodcaller
from datetime import date, datetime

import pytest
import string

import numpy as np

import ibis
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.config as config

from ibis import literal as L

import pandas as pd
import pandas.util.testing as tm

ch = pytest.importorskip('clickhouse_driver')


pytestmark = pytest.mark.clickhouse


# @pytest.yield_fixture
# def guid(con):
#     name = ibis.util.guid()
#     try:
#         yield name
#     finally:
#         con.drop_table(name, force=True)


# @pytest.yield_fixture
# def guid2(con):
#     name = ibis.util.guid()
#     try:
#         yield name
#     finally:
#         con.drop_table(name, force=True)


@pytest.mark.parametrize(('to_type', 'expected'), [
    ('int8', 'CAST(`double_col` AS Int8)'),
    ('int16', 'CAST(`double_col` AS Int16)'),
    ('float', '`double_col`'),
    ('double', 'CAST(`double_col` AS Float64)')
])
def test_cast_double_col(alltypes, translate, to_type, expected):
     expr = alltypes.double_col.cast(to_type)
     assert translate(expr) == expected


@pytest.mark.parametrize(('to_type', 'expected'), [
    ('int8', 'CAST(`string_col` AS Int8)'),
    ('int16', 'CAST(`string_col` AS Int16)'),
    ('string', '`string_col`'),
    ('timestamp', 'CAST(`string_col` AS DateTime)'),
    ('date', 'CAST(`string_col` AS Date)')
])
def test_cast_double_col(alltypes, translate, to_type, expected):
     expr = alltypes.string_col.cast(to_type)
     assert translate(expr) == expected


@pytest.mark.xfail(raises=AssertionError,
                   reason='Clickhouse doesn\'t have decimal type')
def test_decimal_cast():
    assert False


@pytest.mark.parametrize(
    'column',
    [
        'index',
        'Unnamed_0',  # FIXME rename to `Unnamed: 0`
        'id',
        'bool_col',
        'tinyint_col',
        'smallint_col',
        'int_col',
        'bigint_col',
        'float_col',
        'double_col',
        'date_string_col',
        'string_col',
        'timestamp_col',
        'year',
        'month',
    ]
)
def test_noop_cast(alltypes, translate, column):
    col = alltypes[column]
    result = col.cast(col.type())
    assert result.equals(col)
    assert translate(result) == '`{}`'.format(column)


def test_timestamp_cast_noop(alltypes, translate):
    result1 = alltypes.timestamp_col.cast('timestamp')
    result2 = alltypes.int_col.cast('timestamp')

    assert isinstance(result1, ir.TimestampColumn)
    assert isinstance(result2, ir.TimestampColumn)

    assert translate(result1) == '`timestamp_col`'
    assert translate(result2) == 'CAST(`int_col` AS DateTime)'


def test_timestamp_now(con, translate):
    expr = ibis.now()
    # now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    assert translate(expr) == 'now()'
    # assert con.execute(expr) == now


@pytest.mark.parametrize(
    ('func', 'expected'),
    [
        (methodcaller('year'), 2015),
        (methodcaller('month'), 9),
        (methodcaller('day'), 1),
        (methodcaller('hour'), 14),
        (methodcaller('minute'), 48),
        (methodcaller('second'), 5),
    ]
)
def test_simple_datetime_operations(con, func, expected):
    value = ibis.timestamp('2015-09-01 14:48:05.359')
    with pytest.raises(ValueError):
        con.execute(func(value))

    value = ibis.timestamp('2015-09-01 14:48:05')
    con.execute(func(value)) == expected


@pytest.mark.parametrize(
    ('func', 'left', 'right', 'expected'),
    [
        (operator.add, L(3), L(4), 7),
        (operator.sub, L(3), L(4), -1),
        (operator.mul, L(3), L(4), 12),
        (operator.truediv, L(12), L(4), 3),
        (operator.pow, L(12), L(2), 144),
        (operator.mod, L(12), L(5), 2),
        (operator.truediv, L(7), L(2), 3.5),
        (operator.floordiv, L(7), L(2), 3),
        (lambda x, y: x.floordiv(y), L(7), 2, 3),
        (lambda x, y: x.rfloordiv(y), L(2), 7, 3),
    ]
)
def test_binary_arithmetic(con, func, left, right, expected):
    expr = func(left, right)
    result = con.execute(expr)
    assert result == expected


@pytest.mark.parametrize(('op', 'expected'), [
    (lambda a, b: a + b, '`int_col` + `tinyint_col`'),
    (lambda a, b: a - b, '`int_col` - `tinyint_col`'),
    (lambda a, b: a * b, '`int_col` * `tinyint_col`'),
    (lambda a, b: a / b, '`int_col` / `tinyint_col`'),
    (lambda a, b: a ** b, 'pow(`int_col`, `tinyint_col`)'),
    (lambda a, b: a < b, '`int_col` < `tinyint_col`'),
    (lambda a, b: a <= b, '`int_col` <= `tinyint_col`'),
    (lambda a, b: a > b, '`int_col` > `tinyint_col`'),
    (lambda a, b: a >= b, '`int_col` >= `tinyint_col`'),
    (lambda a, b: a == b, '`int_col` = `tinyint_col`'),
    (lambda a, b: a != b, '`int_col` != `tinyint_col`')
])
def test_binary_infix_operators(con, alltypes, translate, op, expected):
    a, b = alltypes.int_col, alltypes.tinyint_col
    expr = op(a, b)
    assert translate(expr) == expected
    assert len(con.execute(expr))


# TODO: test boolean operators
# (h & bool_col, '`h` AND (`a` > 0)'),
# (h | bool_col, '`h` OR (`a` > 0)'),
# (h ^ bool_col, 'xor(`h`, (`a` > 0))')


@pytest.mark.parametrize(('op', 'expected'), [
    (lambda a, b, c: (a + b) + c,
     '(`int_col` + `tinyint_col`) + `double_col`'),
    (lambda a, b, c: a.log() + c,
     'log(`int_col`) + `double_col`'),
    (lambda a, b, c: (b + (-(a + c))),
     '`tinyint_col` + (-(`int_col` + `double_col`))')
])
def test_binary_infix_parenthesization(con, alltypes, translate, op, expected):
    a = alltypes.int_col
    b = alltypes.tinyint_col
    c = alltypes.double_col

    expr = op(a, b, c)
    assert translate(expr) == expected
    assert len(con.execute(expr))


def test_between(con, alltypes, translate):
    expr = alltypes.int_col.between(0, 10)
    assert translate(expr) == '`int_col` BETWEEN 0 AND 10'
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (L('foo_bar'), 'String'),
        (L(5), 'UInt8'),
        (L(1.2345), 'Float64'),
        (L(datetime(2015, 9, 1, hour=14, minute=48, second=5)), 'DateTime'),
        (L(date(2015, 9, 1)), 'Date'),
    ]
)
def test_typeof(con, value, expected):
    assert con.execute(value.typeof()) == expected


@pytest.mark.parametrize(('value', 'expected'), [('foo_bar', 7), ('', 0)])
def test_string_length(con, value, expected):
    assert con.execute(L(value).length()) == expected


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        (methodcaller('substr', 0, 3), 'foo'),
        (methodcaller('substr', 4, 3), 'bar'),
        (methodcaller('substr', 1), 'oo_bar'),
    ]
)
def test_string_substring(con, op, expected):
    value = L('foo_bar')
    assert con.execute(op(value)) == expected


def test_string_column_substring(con, alltypes, translate):
    expr = alltypes.string_col.substr(2)
    assert translate(expr) == 'substring(`string_col`, 2 + 1)'
    assert len(con.execute(expr))

    expr = alltypes.string_col.substr(0, 3)
    assert translate(expr) == 'substring(`string_col`, 0 + 1, 3)'
    assert len(con.execute(expr))


def test_string_reverse(con):
    assert con.execute(L('foo').reverse()) == 'oof'


def test_string_upper(con):
    assert con.execute(L('foo').upper()) == 'FOO'


def test_string_lower(con):
    assert con.execute(L('FOO').lower()) == 'foo'


def test_string_lenght(con):
    assert con.execute(L('FOO').length()) == 3


@pytest.mark.parametrize(
    ('value', 'op', 'expected'),
    [
        (L('foobar'), methodcaller('contains', 'bar'), True),
        (L('foobar'), methodcaller('contains', 'foo'), True),
        (L('foobar'), methodcaller('contains', 'baz'), False),
        (L('100%'), methodcaller('contains', '%'), True),
        (L('a_b_c'), methodcaller('contains', '_'), True),
    ]
)
def test_string_contains(con, op, value, expected):
    assert con.execute(op(value)) == expected


# TODO: clickhouse-driver escaping bug
def test_re_replace(con, translate):
    expr1 = L('Hello, World!').re_replace('.', '\\\\0\\\\0')
    expr2 = L('Hello, World!').re_replace('^', 'here: ')

    assert con.execute(expr1) == 'HHeelllloo,,  WWoorrlldd!!'
    assert con.execute(expr2) == 'here: Hello, World!'


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (L('a'), 0),
        (L('b'), 1),
        (L('d'), -1),  # TODO: what's the expected?
    ]
)
def test_find_in_set(con, value, expected, translate):
    vals = list('abc')
    expr = value.find_in_set(vals)
    assert con.execute(expr) == expected


def test_string_column_find_in_set(con, alltypes, translate):
    s = alltypes.string_col
    vals = list('abc')

    expr = s.find_in_set(vals)
    assert translate(expr) == "indexOf(['a','b','c'], `string_col`) - 1"
    assert len(con.execute(expr))


# def test_parse_url(self):
#     sql = "parse_url(`string_col`, 'HOST')"
#     cases = [
#         (self.table.string_col.parse_url('HOST'), sql)
#     ]
#     self._check_expr_cases(cases)

# def test_string_join(self):
#     cases = [
#         (L(',').join(['a', 'b']), "concat_ws(',', 'a', 'b')")
#     ]
#     self._check_expr_cases(cases)


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L('foobar').find('bar'), 3),
        (L('foobar').find('baz'), -1),

        (L('foobar').like('%bar'), True),
        (L('foobar').like('foo%'), True),
        (L('foobar').like('%baz%'), False),

        (L('foobar').like(['%bar']), True),
        (L('foobar').like(['foo%']), True),
        (L('foobar').like(['%baz%']), False),

        (L('foobar').like(['%bar', 'foo%']), True),

        (L('foobarfoo').replace('foo', 'H'), 'HbarH'),
    ]
)
def test_string_find_like(con, expr, expected):
    assert con.execute(expr) == expected


def test_string_column_like(con, alltypes, translate):
    expr = alltypes.string_col.like('foo%')
    assert translate(expr) == "`string_col` LIKE 'foo%'"
    assert len(con.execute(expr))

    expr = alltypes.string_col.like(['foo%', '%bar'])
    expected = "`string_col` LIKE 'foo%' OR `string_col` LIKE '%bar'"
    assert translate(expr) == expected
    assert len(con.execute(expr))


def test_string_column_find(con, alltypes, translate):
    s = alltypes.string_col

    expr = s.find('a')
    assert translate(expr) == "position(`string_col`, 'a') - 1"
    assert len(con.execute(expr))

    expr = s.find(s)
    assert translate(expr) == "position(`string_col`, `string_col`) - 1"
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('call', 'expected'),
    [
        (methodcaller('log'), 'log(`double_col`)'),
        (methodcaller('log2'), 'log2(`double_col`)'),
        (methodcaller('log10'), 'log10(`double_col`)'),
        (methodcaller('round'), 'round(`double_col`)'),
        (methodcaller('round', 0), 'round(`double_col`, 0)'),
        (methodcaller('round', 2), 'round(`double_col`, 2)'),
        (methodcaller('exp'), 'exp(`double_col`)'),
        (methodcaller('abs'), 'abs(`double_col`)'),
        (methodcaller('ceil'), 'ceil(`double_col`)'),
        (methodcaller('floor'), 'floor(`double_col`)'),
        (methodcaller('sqrt'), 'sqrt(`double_col`)'),
        (methodcaller('sign'), 'intDivOrZero(`double_col`, abs(`double_col`))')
    ]
)
def test_translate_math_functions(con, alltypes, translate, call, expected):
    expr = call(alltypes.double_col)
    assert translate(expr) == expected
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L(-5).abs(), 5),
        (L(5).abs(), 5),
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
)
def test_math_functions(con, expr, expected, translate):
    assert con.execute(expr) == expected


def test_greatest(con, alltypes, translate):
    expr = ibis.greatest(alltypes.int_col, 10)

    assert translate(expr) == "greatest(`int_col`, 10)"
    assert len(con.execute(expr))

    expr = ibis.greatest(alltypes.int_col, alltypes.bigint_col)
    assert translate(expr) == "greatest(`int_col`, `bigint_col`)"
    assert len(con.execute(expr))


def test_least(con, alltypes, translate):
    expr = ibis.least(alltypes.int_col, 10)
    assert translate(expr) == "least(`int_col`, 10)"
    assert len(con.execute(expr))

    expr = ibis.least(alltypes.int_col, alltypes.bigint_col)
    assert translate(expr) == "least(`int_col`, `bigint_col`)"
    assert len(con.execute(expr))


# TODO: clickhouse-driver escaping bug
@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L('abcd').re_search('[a-z]'), True),
        (L('abcd').re_search('[\\\d]+'), False),
        (L('1222').re_search('[\\\d]+'), True),
    ]
)
def test_regexp(con, expr, expected):
    assert con.execute(expr) == expected


# TODO: two cases are failing
@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L('abcd').re_extract('([a-z]+)', 0), 'abcd'),
        # (L('abcd').re_extract('(ab)(cd)', 1), 'cd'),

        # valid group number but no match => empty string
        (L('abcd').re_extract('(\\\d)', 0), ''),

        # match but not a valid group number => NULL
        # (L('abcd').re_extract('abcd', 3), None),
    ]
)
def test_regexp_extract(con, expr, expected, translate):
    assert con.execute(expr) == expected


def test_column_regexp_extract(con, alltypes, translate):
    expected = "extractAll(`string_col`, '[\d]+')[3 + 1]"

    expr = alltypes.string_col.re_extract('[\d]+', 3)
    assert translate(expr) == expected
    assert len(con.execute(expr))


def test_column_regexp_replace(con, alltypes, translate):
    expected = "replaceRegexpAll(`string_col`, '[\d]+', 'aaa')"

    expr = alltypes.string_col.re_replace('[\d]+', 'aaa')
    assert translate(expr) == expected
    assert len(con.execute(expr))


@pytest.mark.parametrize('column', ['int_col', 'float_col', 'bool_col'])
def test_negate(con, alltypes, translate, column):
    # clickhouse represent boolean as UInt8
    # TODO: how should I identify boolean cols?

    expr = -getattr(alltypes, column)
    assert translate(expr) == '-`{0}`'.format(column)
    assert len(con.execute(expr))


def test_field_in_literals(con, alltypes, translate):
    expr = alltypes.string_col.isin(['foo', 'bar', 'baz'])
    assert translate(expr) == "`string_col` IN ('foo', 'bar', 'baz')"
    assert len(con.execute(expr))

    expr = alltypes.string_col.notin(['foo', 'bar', 'baz'])
    assert translate(expr) == "`string_col` NOT IN ('foo', 'bar', 'baz')"
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('reduction', 'func_translated'),
    [
        ('sum', 'sum'),
        ('count', 'count'),
        ('mean', 'avg'),
        ('max', 'max'),
        ('min', 'min'),
        ('std', 'stddevSamp'),
        ('var', 'varSamp')
    ]
)
def test_reduction_where(con, alltypes, translate, reduction, func_translated):
    template = '{0}If(`double_col`, `bigint_col` < 70)'
    expected = template.format(func_translated)

    method = getattr(alltypes.double_col, reduction)
    cond = alltypes.bigint_col < 70
    expr = method(where=cond)

    assert translate(expr) == expected
    assert isinstance(con.execute(expr), (np.float, np.uint))


def test_std_var_pop(con, alltypes, translate):
    cond = alltypes.bigint_col < 70
    expr1 = alltypes.double_col.std(where=cond, how='pop')
    expr2 = alltypes.double_col.var(where=cond, how='pop')

    assert translate(expr1) == 'stddevPopIf(`double_col`, `bigint_col` < 70)'
    assert translate(expr2) == 'varPopIf(`double_col`, `bigint_col` < 70)'
    assert isinstance(con.execute(expr1), np.float)
    assert isinstance(con.execute(expr2), np.float)


@pytest.mark.parametrize('reduction', ['sum', 'count', 'max', 'min'])
def test_reduction_invalid_where(con, alltypes, reduction):
    condbad_literal = L('T')

    with pytest.raises(TypeError):
        fn = methodcaller(reduction, where=condbad_literal)
        expr = fn(alltypes.double_col)


# def test_hash(self):
#     expr = self.table.int_col.hash()
#     assert isinstance(expr, ir.Int64Column)
#     assert isinstance(self.table.int_col.sum().hash(),
#                       ir.Int64Scalar)

#     cases = [
#         (self.table.int_col.hash(), 'fnv_hash(`int_col`)')
#     ]
#     self._check_expr_cases(cases)

# @pytest.mark.parametrize(
#     ('expr', 'expected'),
#     [
#         (ibis.NA.fillna(5), 5),
#         (L(5).fillna(10), 5),
#         (L(5).nullif(5), None),
#         (L(10).nullif(5), 10),
#     ]
# )
# def test_fillna_nullif(con, expr, expected):
#     assert con.execute(expr) == expected


# @pytest.mark.parametrize(
#     ('expr', 'expected'),
#     [
#         (ibis.coalesce(ibis.NA, ibis.NA), None),
#         (ibis.coalesce(ibis.NA, ibis.NA, ibis.NA.cast('double')), None),
#         (
#             ibis.coalesce(
#                 ibis.NA.cast('int8'),
#                 ibis.NA.cast('int8'),
#                 ibis.NA.cast('int8'),
#             ),
#             None,
#         ),
#     ]
# )
# def test_coalesce_all_na(con, expr, expected):
#     assert con.execute(expr) == expected


def test_numeric_builtins_work(con, alltypes, df, translate):
    expr = alltypes.double_col
    result = expr.execute()
    expected = df.double_col.fillna(0)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('op', 'pandas_op'),
    [
        (
            lambda t: (t.double_col > 20).ifelse(10, -20),
            lambda df: pd.Series(np.where(df.double_col > 20, 10, -20),
                                 dtype='int16')
        ),
        (
            lambda t: (t.double_col > 20).ifelse(10, -20).abs(),
            lambda df: (pd.Series(np.where(df.double_col > 20, 10, -20))
                          .abs()
                          .astype('uint16'))
        ),
    ]
)
def test_ifelse(alltypes, df, op, pandas_op, translate):
    expr = op(alltypes)
    result = expr.execute()
    result.name = None
    expected = pandas_op(df)

    tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize(
#     ('func', 'pandas_func'),
#     [
#         # tier and histogram
#         (
#             lambda d: d.bucket([0, 10, 25, 50, 100]),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50, 100], right=False, labels=False,
#             )
#         ),
#         (
#             lambda d: d.bucket([0, 10, 25, 50], include_over=True),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50, np.inf], right=False, labels=False
#             )
#         ),
#         (
#             lambda d: d.bucket([0, 10, 25, 50], close_extreme=False),
#             lambda s: pd.cut(s, [0, 10, 25, 50], right=False, labels=False),
#         ),
#         (
#             lambda d: d.bucket(
#                 [0, 10, 25, 50], closed='right', close_extreme=False
#             ),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50],
#                 include_lowest=False,
#                 right=True,
#                 labels=False,
#             )
#         ),
#         (
#             lambda d: d.bucket([10, 25, 50, 100], include_under=True),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50, 100], right=False, labels=False
#             ),
#         ),
#     ]
# )
# def test_bucket(alltypes, df, func, pandas_func):
#     expr = func(alltypes.double_col)
#     result = expr.execute()
#     expected = pandas_func(df.double_col)
#     tm.assert_series_equal(result, expected, check_names=False)


# def test_category_label(alltypes, df):
#     t = alltypes
#     d = t.double_col

#     bins = [0, 10, 25, 50, 100]
#     labels = ['a', 'b', 'c', 'd']
#     bucket = d.bucket(bins)
#     expr = bucket.label(labels)
#     result = expr.execute().astype('category', ordered=True)
#     result.name = 'double_col'

#     expected = pd.cut(df.double_col, bins, labels=labels, right=False)

#     tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('func', 'pandas_func'),
    [
        (
            lambda t, cond: t.bool_col.count(),
            lambda df, cond: df.bool_col.count(),
        ),
        (
            lambda t, cond: t.bool_col.nunique(),
            lambda df, cond: df.bool_col.nunique(),
        ),
        (
            lambda t, cond: t.bool_col.approx_nunique(),
            lambda df, cond: df.bool_col.nunique(),
        ),
        # group_concat
        # (
        #     lambda t, cond: t.bool_col.any(),
        #     lambda df, cond: df.bool_col.any(),
        # ),
        # (
        #     lambda t, cond: t.bool_col.all(),
        #     lambda df, cond: df.bool_col.all(),
        # ),
        # (
        #     lambda t, cond: t.bool_col.notany(),
        #     lambda df, cond: ~df.bool_col.any(),
        # ),
        # (
        #     lambda t, cond: t.bool_col.notall(),
        #     lambda df, cond: ~df.bool_col.all(),
        # ),
        (
            lambda t, cond: t.double_col.sum(),
            lambda df, cond: df.double_col.sum(),
        ),
        (
            lambda t, cond: t.double_col.mean(),
            lambda df, cond: df.double_col.mean(),
        ),
        (
            lambda t, cond: t.int_col.approx_median(),
            lambda df, cond: df.int_col.median(),
        ),
        (
            lambda t, cond: t.double_col.min(),
            lambda df, cond: df.double_col.min(),
        ),
        (
            lambda t, cond: t.double_col.max(),
            lambda df, cond: df.double_col.max(),
        ),
        (
            lambda t, cond: t.double_col.var(),
            lambda df, cond: df.double_col.var(),
        ),
        (
            lambda t, cond: t.double_col.std(),
            lambda df, cond: df.double_col.std(),
        ),
        (
            lambda t, cond: t.double_col.var(how='sample'),
            lambda df, cond: df.double_col.var(ddof=1),
        ),
        (
            lambda t, cond: t.double_col.std(how='pop'),
            lambda df, cond: df.double_col.std(ddof=0),
        ),
        (
            lambda t, cond: t.bool_col.count(where=cond),
            lambda df, cond: df.bool_col[cond].count(),
        ),
        (
            lambda t, cond: t.bool_col.nunique(where=cond),
            lambda df, cond: df.bool_col[cond].nunique(),
        ),
        (
            lambda t, cond: t.bool_col.approx_nunique(where=cond),
            lambda df, cond: df.bool_col[cond].nunique(),
        ),
        (
            lambda t, cond: t.double_col.sum(where=cond),
            lambda df, cond: df.double_col[cond].sum(),
        ),
        (
            lambda t, cond: t.double_col.mean(where=cond),
            lambda df, cond: df.double_col[cond].mean(),
        ),
        (
            lambda t, cond: t.int_col.approx_median(where=cond),
            lambda df, cond: df.int_col[cond].median(),
        ),
        (
            lambda t, cond: t.double_col.min(where=cond),
            lambda df, cond: df.double_col[cond].min(),
        ),
        (
            lambda t, cond: t.double_col.max(where=cond),
            lambda df, cond: df.double_col[cond].max(),
        ),
        (
            lambda t, cond: t.double_col.var(where=cond),
            lambda df, cond: df.double_col[cond].var(),
        ),
        (
            lambda t, cond: t.double_col.std(where=cond),
            lambda df, cond: df.double_col[cond].std(),
        ),
        (
            lambda t, cond: t.double_col.var(where=cond, how='sample'),
            lambda df, cond: df.double_col[cond].var(),
        ),
        (
            lambda t, cond: t.double_col.std(where=cond, how='pop'),
            lambda df, cond: df.double_col[cond].std(ddof=0),
        ),
    ]
)
def test_aggregations(alltypes, df, func, pandas_func, translate):
    table = alltypes.limit(100)
    count = table.count().execute()
    df = df.head(int(count))

    cond = table.string_col.isin(['1', '7'])
    mask = cond.execute().astype('bool')
    expr = func(table, cond)

    result = expr.execute()
    expected = pandas_func(df, mask)

    np.testing.assert_allclose(result, expected)


# def test_group_concat(alltypes, df):
#     expr = alltypes.string_col.group_concat()
#     result = expr.execute()
#     expected = ','.join(df.string_col.dropna())
#     assert result == expected


def test_distinct_aggregates(alltypes, df, translate):
    expr = alltypes.limit(100).double_col.nunique()
    result = expr.execute()

    assert translate(expr) == 'uniq(`double_col`)'
    assert result == df.head(100).double_col.nunique()


# def test_not_exists(alltypes, df):
#     t = alltypes
#     t2 = t.view()

#     expr = t[~(t.string_col == t2.string_col).any()]
#     result = expr.execute()

#     left, right = df, t2.execute()
#     expected = left[left.string_col != right.string_col]

#     tm.assert_frame_equal(
#         result, expected,
#         check_index_type=False,
#         check_dtype=False,
#     )


# def test_interactive_repr_shows_error(alltypes):
#     # #591. Doing this in PostgreSQL because so many built-in functions are
#     # not available

#     expr = alltypes.double_col.approx_median()

#     with config.option_context('interactive', True):
#         result = repr(expr)

#     assert 'no translator rule' in result.lower()


# def test_null_column(alltypes):
#     t = alltypes
#     nrows = t.count().execute()
#     expr = t.mutate(na_column=ibis.NA).na_column
#     result = expr.execute()
#     tm.assert_series_equal(
#         result,
#         pd.Series([None] * nrows, name='na_column')
#     )


# def test_null_column_union(alltypes, df):
#     t = alltypes
#     s = alltypes[['double_col']].mutate(
#         string_col=ibis.NA.cast('string'),
#     )
#     expr = t[['double_col', 'string_col']].union(s)
#     result = expr.execute()
#     nrows = t.count().execute()
#     expected = pd.concat(
#         [
#             df[['double_col', 'string_col']],
#             pd.concat(
#                 [
#                     df[['double_col']],
#                     pd.DataFrame({'string_col': [None] * nrows})
#                 ],
#                 axis=1,
#             )
#         ],
#         axis=0,
#         ignore_index=True
#     )
#     tm.assert_frame_equal(result, expected)


# def test_anonymous_aggregate(alltypes, df):
#     t = alltypes
#     expr = t[t.double_col > t.double_col.mean()]
#     result = expr.execute()
#     expected = df[df.double_col > df.double_col.mean()].reset_index(
#         drop=True
#     )
#     tm.assert_frame_equal(result, expected)


# def test_identical_to(con, df):
#     # TODO: abstract this testing logic out into parameterized fixtures
#     t = con.table('functional_alltypes')
#     dt = df[['tinyint_col', 'double_col']]
#     expr = t.tinyint_col.identical_to(t.double_col)
#     result = expr.execute()
#     expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
#         dt.tinyint_col == dt.double_col
#     )
#     expected.name = result.name
#     tm.assert_series_equal(result, expected)


# def test_rank(con):
#     t = con.table('functional_alltypes')
#     expr = t.double_col.rank()
#     sqla_expr = expr.compile()
#     result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
#     expected = """\
# SELECT rank() OVER (ORDER BY t0.double_col ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS tmp
# FROM functional_alltypes AS t0"""  # noqa: E501,W291
#     assert result == expected


# def test_percent_rank(con):
#     t = con.table('functional_alltypes')
#     expr = t.double_col.percent_rank()
#     sqla_expr = expr.compile()
#     result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
#     expected = """\
# SELECT percent_rank() OVER (ORDER BY t0.double_col ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS tmp
# FROM functional_alltypes AS t0"""  # noqa: E501,W291
#     assert result == expected


# def test_ntile(con):
#     t = con.table('functional_alltypes')
#     expr = t.double_col.ntile(7)
#     sqla_expr = expr.compile()
#     result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
#     expected = """\
# SELECT ntile(7) OVER (ORDER BY t0.double_col ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - 1 AS tmp
# FROM functional_alltypes AS t0"""  # noqa: E501,W291
#     assert result == expected


# @pytest.mark.parametrize('op', [operator.invert, operator.neg])
# def test_not_and_negate_bool(con, op, df):
#     t = con.table('functional_alltypes').limit(10)
#     expr = t.projection([op(t.bool_col).name('bool_col')])
#     result = expr.execute().bool_col
#     expected = op(df.head(10).bool_col)
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize(
#     'field',
#     [
#         'tinyint_col',
#         'smallint_col',
#         'int_col',
#         'bigint_col',
#         'float_col',
#         'double_col',
#         'year',
#         'month',
#     ]
# )
# def test_negate_non_boolean(con, field, df):
#     t = con.table('functional_alltypes').limit(10)
#     expr = t.projection([(-t[field]).name(field)])
#     result = expr.execute()[field]
#     expected = -df.head(10)[field]
#     tm.assert_series_equal(result, expected)


# def test_negate_boolean(con, df):
#     t = con.table('functional_alltypes').limit(10)
#     expr = t.projection([(-t.bool_col).name('bool_col')])
#     result = expr.execute().bool_col
#     expected = -df.head(10).bool_col
#     tm.assert_series_equal(result, expected)


# @pytest.mark.parametrize(
#     ('attr', 'expected'),
#     [
#         (operator.methodcaller('year'), {2009, 2010}),
#         (operator.methodcaller('month'), set(range(1, 13))),
#         (operator.methodcaller('day'), set(range(1, 32)))
#     ]
# )
# def test_date_extract_field(db, attr, expected):
#     t = db.functional_alltypes
#     expr = attr(t.timestamp_col.cast('date')).distinct()
#     result = expr.execute().astype(int)
#     assert set(result) == expected


# @pytest.mark.parametrize(
#     'op',
#     list(map(
#         operator.methodcaller, ['sum', 'mean', 'min', 'max', 'std', 'var']
#     ))
# )
# def test_boolean_reduction(alltypes, op, df):
#     result = op(alltypes.bool_col).execute()
#     assert result == op(df.bool_col)


# def test_boolean_summary(alltypes):
#     expr = alltypes.bool_col.summary()
#     result = expr.execute()
#     expected = pd.DataFrame(
#         [[7300, 0, 0, 1, 3650, 0.5, 2]],
#         columns=[
#             'count',
#             'nulls',
#             'min',
#             'max',
#             'sum',
#             'mean',
#             'approx_nunique',
#         ]
#     )
#     tm.assert_frame_equal(result, expected)


# def test_timestamp_with_timezone(con):
#     t = con.table('tzone')
#     result = t.ts.execute()
#     assert str(result.dtype.tz)


# @pytest.fixture(
#     params=[
#         None,
#         'UTC',
#         'America/New_York',
#         'America/Los_Angeles',
#         'Europe/Paris',
#         'Chile/Continental',
#         'Asia/Tel_Aviv',
#         'Asia/Tokyo',
#         'Africa/Nairobi',
#         'Australia/Sydney',
#     ]
# )
# def tz(request):
#     return request.param


# @pytest.yield_fixture
# def tzone_compute(con, guid, tz):
#     schema = ibis.schema([
#         ('ts', dt.timestamp(tz)),
#         ('b', 'double'),
#         ('c', 'string'),
#     ])
#     con.create_table(guid, schema=schema)
#     t = con.table(guid)

#     n = 10
#     df = pd.DataFrame({
#         'ts': pd.date_range('2017-04-01', periods=n, tz=tz).values,
#         'b': np.arange(n).astype('float64'),
#         'c': list(string.ascii_lowercase[:n]),
#     })

#     df.to_sql(
#         guid,
#         con.con,
#         index=False,
#         if_exists='append',
#         dtype={
#             'ts': sa.TIMESTAMP(timezone=True),
#             'b': sa.FLOAT,
#             'c': sa.TEXT,
#         }
#     )

#     try:
#         yield t
#     finally:
#         con.drop_table(guid)
#         assert guid not in con.list_tables()


# def test_ts_timezone_is_preserved(tzone_compute, tz):
#     assert dt.Timestamp(tz).equals(tzone_compute.ts.type())


# def test_timestamp_with_timezone_select(tzone_compute, tz):
#     ts = tzone_compute.ts.execute()
#     assert str(getattr(ts.dtype, 'tz', None)) == str(tz)


# def test_timestamp_type_accepts_all_timezones(con):
#     assert all(
#         dt.Timestamp(row.name).timezone == row.name
#         for row in con.con.execute(
#             'SELECT name FROM pg_timezone_names'
#         )
#     )


# @pytest.mark.parametrize(
#     ('left', 'right', 'type'),
#     [
#         (L('2017-04-01'), date(2017, 4, 2), dt.date),
#         (date(2017, 4, 2), L('2017-04-01'), dt.date),
#         (
#             L('2017-04-01 01:02:33'),
#             datetime(2017, 4, 1, 1, 3, 34),
#             dt.timestamp
#         ),
#         (
#             datetime(2017, 4, 1, 1, 3, 34),
#             L('2017-04-01 01:02:33'),
#             dt.timestamp
#         ),
#     ]
# )
# @pytest.mark.parametrize(
#     'op',
#     [
#         operator.eq,
#         operator.ne,
#         operator.lt,
#         operator.le,
#         operator.gt,
#         operator.ge,
#     ]
# )
# def test_string_temporal_compare(con, op, left, right, type):
#     expr = op(left, right)
#     result = con.execute(expr)
#     left_raw = con.execute(L(left).cast(type))
#     right_raw = con.execute(L(right).cast(type))
#     expected = op(left_raw, right_raw)
#     assert result == expected


# @pytest.mark.parametrize(
#     ('left', 'right'),
#     [
#         (L('2017-03-31').cast(dt.date), date(2017, 4, 2)),
#         (date(2017, 3, 31), L('2017-04-02').cast(dt.date)),
#         (
#             L('2017-03-31 00:02:33').cast(dt.timestamp),
#             datetime(2017, 4, 1, 1, 3, 34),
#         ),
#         (
#             datetime(2017, 3, 31, 0, 2, 33),
#             L('2017-04-01 01:03:34').cast(dt.timestamp),
#         ),
#     ]
# )
# @pytest.mark.parametrize(
#     'op',
#     [
#         lambda left, right: ibis.timestamp('2017-04-01 00:02:34').between(
#             left, right
#         ),
#         lambda left, right: ibis.timestamp('2017-04-01').cast(dt.date).between(
#             left, right
#         ),
#     ]
# )
# def test_string_temporal_compare_between(con, op, left, right):
#     expr = op(left, right)
#     result = con.execute(expr)
#     assert isinstance(result, (bool, np.bool_))
#     assert result
