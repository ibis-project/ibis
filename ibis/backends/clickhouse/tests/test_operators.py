from __future__ import annotations

import operator
from datetime import date, datetime

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis import literal as L

pytest.importorskip("clickhouse_connect")


@pytest.mark.parametrize(
    ("left", "right", "type"),
    [
        (L("2017-04-01"), date(2017, 4, 2), dt.date),
        (date(2017, 4, 2), L("2017-04-01"), dt.date),
        (
            L("2017-04-01 01:02:33"),
            datetime(2017, 4, 1, 1, 3, 34),
            dt.timestamp,
        ),
        (
            datetime(2017, 4, 1, 1, 3, 34),
            L("2017-04-01 01:02:33"),
            dt.timestamp,
        ),
    ],
)
@pytest.mark.parametrize(
    "op",
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
    "op",
    ["add", "sub", "mul", "truediv", "pow", "lt", "le", "gt", "ge", "eq", "ne"],
)
def test_binary_infix_operators(con, alltypes, op, assert_sql):
    func = getattr(operator, op)
    a, b = alltypes.int_col, alltypes.tinyint_col
    expr = func(a, b)
    assert_sql(expr)
    assert len(con.execute(expr))


# TODO: test boolean operators
# (h & bool_col, '`h` AND (`a` > 0)'),
# (h | bool_col, '`h` OR (`a` > 0)'),
# (h ^ bool_col, 'xor(`h`, (`a` > 0))')


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b, c: (a + b) + c,
        lambda a, _, c: a.log() + c,
        lambda a, b, c: (b + (-(a + c))),
    ],
)
def test_binary_infix_parenthesization(con, alltypes, op, assert_sql):
    a = alltypes.int_col
    b = alltypes.tinyint_col
    c = alltypes.double_col

    expr = op(a, b, c)
    assert_sql(expr)
    assert len(con.execute(expr))


def test_between(con, alltypes, assert_sql):
    expr = alltypes.int_col.between(0, 10)
    assert_sql(expr)
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (L("2017-03-31").cast(dt.date), date(2017, 4, 2)),
        (date(2017, 3, 31), L("2017-04-02").cast(dt.date)),
    ],
)
def test_string_temporal_compare_between_dates(con, left, right):
    expr = ibis.timestamp("2017-04-01").cast(dt.date).between(left, right)
    result = con.execute(expr)
    assert result


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            L("2017-03-31 00:02:33").cast(dt.timestamp),
            datetime(2017, 4, 1, 1, 3, 34),
        ),
        (
            datetime(2017, 3, 31, 0, 2, 33),
            L("2017-04-01 01:03:34").cast(dt.timestamp),
        ),
    ],
)
def test_string_temporal_compare_between_datetimes(con, left, right):
    expr = ibis.timestamp("2017-04-01 00:02:34").between(left, right)
    result = con.execute(expr)
    assert result


@pytest.mark.parametrize("container", [list, tuple, set])
def test_field_in_literals(con, alltypes, df, container):
    values = {"1", "2", "3", "5", "7"}
    foobar = container(values)

    expr = alltypes.string_col.isin(foobar)
    result_col = con.execute(expr.name("result"))
    expected_col = df.string_col.isin(foobar).rename("result")
    tm.assert_series_equal(result_col, expected_col)

    expr = alltypes.string_col.notin(foobar)
    result_col = con.execute(expr.name("result"))
    expected_col = ~df.string_col.isin(foobar).rename("result")
    tm.assert_series_equal(result_col, expected_col)


@pytest.mark.parametrize("column", ["int_col", "float_col"])
def test_negate(con, alltypes, column, assert_sql):
    expr = -alltypes[column]
    assert_sql(expr)
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    "field",
    [
        "tinyint_col",
        "smallint_col",
        "int_col",
        "bigint_col",
        "float_col",
        "double_col",
        "year",
        "month",
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
    ("op", "pandas_op"),
    [
        (
            lambda t: (t.double_col > 20).ifelse(10, -20),
            lambda df: pd.Series(np.where(df.double_col > 20, 10, -20), dtype="int8"),
        ),
        (
            lambda t: (t.double_col > 20).ifelse(10, -20).abs(),
            lambda df: (
                pd.Series(np.where(df.double_col > 20, 10, -20)).abs().astype("int8")
            ),
        ),
    ],
)
def test_ifelse(alltypes, df, op, pandas_op):
    expr = op(alltypes)
    result = expr.execute()
    result.name = None
    expected = pandas_op(df)

    tm.assert_series_equal(result, expected)


def test_simple_case(con, alltypes, assert_sql):
    t = alltypes
    expr = t.string_col.cases(("foo", "bar"), ("baz", "qux"), else_="default")

    assert_sql(expr)
    assert len(con.execute(expr))


def test_search_case(con, alltypes, assert_sql):
    t = alltypes
    expr = ibis.cases(
        (t.float_col > 0, t.int_col * 2),
        (t.float_col < 0, t.int_col),
        else_=0,
    )

    assert_sql(expr)
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    "arr",
    [
        [1, 2, 3],
        ["qw", "wq", "1"],
        [1.2, 0.3, 0.4],
        [[1], [1, 2], [1, 2, 3]],
    ],
)
@pytest.mark.parametrize(
    "gen_idx",
    [
        lambda arr: range(len(arr)),
        lambda arr: range(-len(arr), 0),
    ],
    ids=["positive", "negative"],
)
def test_array_index(con, arr, gen_idx):
    expr = L(arr)
    for i in gen_idx(arr):
        el_expr = expr[i]
        el = con.execute(el_expr)
        assert el == arr[i]


@pytest.mark.parametrize(
    "arrays",
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
    expected = sum(arrays, [])  # noqa: RUF017
    for arr in arrays:
        expr += L(arr, type="!array<int8>")

    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("arr", "times"),
    [([1], 1), ([1], 2), ([1], 3), ([1, 2], 1), ([1, 2], 2), ([1, 2], 3)],
)
def test_array_repeat(con, arr, times):
    expected = arr * times
    expr = L(arr) * times
    assert con.execute(expr) == expected


@pytest.mark.parametrize("arr", [[], [1], [1, 2, 3, 4, 5, 6]])
@pytest.mark.parametrize("start", [None, 0, 1, 2, -1, -3])
@pytest.mark.parametrize("stop", [None, 0, 1, 3, -2, -4])
def test_array_slice(con, arr, start, stop):
    expr = L(arr, type="array<int8>")
    assert con.execute(expr[start:stop]) == arr[start:stop]
