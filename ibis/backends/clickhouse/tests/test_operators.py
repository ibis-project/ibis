from __future__ import annotations

import operator
from datetime import date, datetime

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import literal as L

pytest.importorskip("clickhouse_connect")


@pytest.mark.parametrize(
    ('left', 'right', 'type'),
    [
        (L('2017-04-01'), date(2017, 4, 2), dt.date),
        (date(2017, 4, 2), L('2017-04-01'), dt.date),
        (
            L('2017-04-01 01:02:33'),
            datetime(2017, 4, 1, 1, 3, 34),
            dt.timestamp,
        ),
        (
            datetime(2017, 4, 1, 1, 3, 34),
            L('2017-04-01 01:02:33'),
            dt.timestamp,
        ),
    ],
)
@pytest.mark.parametrize(
    'op',
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
def test_string_temporal_compare(con, op, left, right, type):
    expr = op(left, right)
    result = con.execute(expr)
    left_raw = con.execute(L(left).cast(type))
    right_raw = con.execute(L(right).cast(type))
    expected = op(left_raw, right_raw)
    assert result == expected


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        (lambda a, b: a + b, 'int_col + tinyint_col'),
        (lambda a, b: a - b, 'int_col - tinyint_col'),
        (lambda a, b: a * b, 'int_col * tinyint_col'),
        (lambda a, b: a / b, 'int_col / tinyint_col'),
        (lambda a, b: a**b, 'pow(int_col, tinyint_col)'),
        (lambda a, b: a < b, 'int_col < tinyint_col'),
        (lambda a, b: a <= b, 'int_col <= tinyint_col'),
        (lambda a, b: a > b, 'int_col > tinyint_col'),
        (lambda a, b: a >= b, 'int_col >= tinyint_col'),
        (lambda a, b: a == b, 'int_col = tinyint_col'),
        (lambda a, b: a != b, 'int_col != tinyint_col'),
    ],
)
def test_binary_infix_operators(con, alltypes, translate, op, expected):
    a, b = alltypes.int_col, alltypes.tinyint_col
    expr = op(a, b)
    assert translate(expr.op()) == expected
    assert len(con.execute(expr))


# TODO: test boolean operators
# (h & bool_col, '`h` AND (`a` > 0)'),
# (h | bool_col, '`h` OR (`a` > 0)'),
# (h ^ bool_col, 'xor(`h`, (`a` > 0))')


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        (
            lambda a, b, c: (a + b) + c,
            '(int_col + tinyint_col) + double_col',
        ),
        (lambda a, _, c: a.log() + c, 'log(int_col) + double_col'),
        (
            lambda a, b, c: (b + (-(a + c))),
            'tinyint_col + (-(int_col + double_col))',
        ),
    ],
)
def test_binary_infix_parenthesization(con, alltypes, translate, op, expected):
    a = alltypes.int_col
    b = alltypes.tinyint_col
    c = alltypes.double_col

    expr = op(a, b, c)
    assert translate(expr.op()) == expected
    assert len(con.execute(expr))


def test_between(con, alltypes, translate):
    expr = alltypes.int_col.between(0, 10)
    assert translate(expr.op()) == 'int_col BETWEEN 0 AND 10'
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (L('2017-03-31').cast(dt.date), date(2017, 4, 2)),
        (date(2017, 3, 31), L('2017-04-02').cast(dt.date)),
    ],
)
def test_string_temporal_compare_between_dates(con, left, right):
    expr = ibis.timestamp('2017-04-01').cast(dt.date).between(left, right)
    result = con.execute(expr)
    assert result


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (
            L('2017-03-31 00:02:33').cast(dt.timestamp),
            datetime(2017, 4, 1, 1, 3, 34),
        ),
        (
            datetime(2017, 3, 31, 0, 2, 33),
            L('2017-04-01 01:03:34').cast(dt.timestamp),
        ),
    ],
)
def test_string_temporal_compare_between_datetimes(con, left, right):
    expr = ibis.timestamp('2017-04-01 00:02:34').between(left, right)
    result = con.execute(expr)
    assert result


@pytest.mark.parametrize('container', [list, tuple, set])
def test_field_in_literals(con, alltypes, translate, container):
    values = {'foo', 'bar', 'baz'}
    foobar = container(values)
    expected = tuple(values)

    expr = alltypes.string_col.isin(foobar)
    assert translate(expr.op()) == f"string_col IN {expected}"
    assert len(con.execute(expr))

    expr = alltypes.string_col.notin(foobar)
    assert translate(expr.op()) == f"string_col NOT IN {expected}"
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ("column", "operator"),
    [
        param("int_col", "-", id="int_col"),
        param("float_col", "-", id="float_col"),
        param("bool_col", "NOT ", id="bool_col"),
    ],
)
def test_negate(con, alltypes, translate, column, operator):
    expr = -alltypes[column]
    assert translate(expr.op()) == f"{operator}{column}"
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    'field',
    [
        'tinyint_col',
        'smallint_col',
        'int_col',
        'bigint_col',
        'float_col',
        'double_col',
        'year',
        'month',
    ],
)
def test_negate_non_boolean(alltypes, field, df):
    t = alltypes.limit(10)
    expr = t.select((-t[field]).name(field))
    result = expr.execute()[field]
    expected = -df.head(10)[field]
    tm.assert_series_equal(result, expected)


def test_negate_literal(con):
    expr = -L(5.245)
    assert round(con.execute(expr), 3) == -5.245


@pytest.mark.parametrize(
    ('op', 'pandas_op'),
    [
        (
            lambda t: (t.double_col > 20).ifelse(10, -20),
            lambda df: pd.Series(np.where(df.double_col > 20, 10, -20), dtype='int8'),
        ),
        (
            lambda t: (t.double_col > 20).ifelse(10, -20).abs(),
            lambda df: (
                pd.Series(np.where(df.double_col > 20, 10, -20)).abs().astype('int8')
            ),
        ),
    ],
)
def test_ifelse(alltypes, df, op, pandas_op, translate):
    expr = op(alltypes)
    result = expr.execute()
    result.name = None
    expected = pandas_op(df)

    tm.assert_series_equal(result, expected)


def test_simple_case(con, alltypes, translate):
    t = alltypes
    expr = (
        t.string_col.case().when('foo', 'bar').when('baz', 'qux').else_('default').end()
    )

    expected = """CASE string_col WHEN 'foo' THEN 'bar' WHEN 'baz' THEN 'qux' ELSE 'default' END"""
    assert translate(expr.op()) == expected
    assert len(con.execute(expr))


def test_search_case(con, alltypes, translate):
    t = alltypes
    expr = (
        ibis.case()
        .when(t.float_col > 0, t.int_col * 2)
        .when(t.float_col < 0, t.int_col)
        .else_(0)
        .end()
    )

    expected = """CASE WHEN float_col > 0 THEN int_col * 2 WHEN float_col < 0 THEN int_col ELSE 0 END"""
    assert translate(expr.op()) == expected
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    'arr',
    [
        [1, 2, 3],
        ['qw', 'wq', '1'],
        [1.2, 0.3, 0.4],
        [[1], [1, 2], [1, 2, 3]],
    ],
)
@pytest.mark.parametrize(
    'ids',
    [
        lambda arr: range(len(arr)),
        lambda arr: range(-len(arr), 0),
    ],
)
def test_array_index(con, arr, ids):
    expr = L(arr)
    for i in ids(arr):
        el_expr = expr[i]
        el = con.execute(el_expr)
        assert el == arr[i]


@pytest.mark.parametrize(
    'arrays',
    [
        ([1], [2]),
        ([1], [1, 2]),
        ([1, 2], [1]),
        ([1, 2], [3, 4]),
        ([1, 2], [3, 4], [5, 6]),
    ],
)
def test_array_concat(con, arrays):
    expr = L([]).cast("!array<int8>")
    expected = sum(arrays, [])
    for arr in arrays:
        expr += L(arr, type="!array<int8>")

    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('arr', 'times'),
    [([1], 1), ([1], 2), ([1], 3), ([1, 2], 1), ([1, 2], 2), ([1, 2], 3)],
)
def test_array_repeat(con, arr, times):
    expected = arr * times
    expr = L(arr) * times
    assert con.execute(expr) == expected


@pytest.mark.parametrize('arr', [[], [1], [1, 2, 3, 4, 5, 6]])
@pytest.mark.parametrize('start', [None, 0, 1, 2, -1, -3])
@pytest.mark.parametrize('stop', [None, 0, 1, 3, -2, -4])
def test_array_slice(con, arr, start, stop):
    expr = L(arr, type="array<int8>")
    assert con.execute(expr[start:stop]) == arr[start:stop]
