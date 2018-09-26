import pytest

import numpy as np

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.types as ir
import ibis.expr.datatypes as dt

from ibis.pandas.udf import udf, nullable
from ibis.pandas.dispatch import pause_ordering


with pause_ordering():

    @udf.elementwise(input_type=[dt.string], output_type=dt.int64)
    def my_string_length(series, **kwargs):
        return series.str.len() * 2

    @udf.elementwise(input_type=[dt.double, dt.double], output_type=dt.double)
    def my_add(series1, series2, *kwargs):
        return series1 + series2

    @udf.reduction(input_type=[dt.string], output_type=dt.int64)
    def my_string_length_sum(series, **kwargs):
        return (series.str.len() * 2).sum()

    @udf.reduction(input_type=[dt.double, dt.double], output_type=dt.double)
    def my_corr(lhs, rhs, **kwargs):
        return lhs.corr(rhs)

    @udf.elementwise([dt.double], dt.double)
    def add_one(x):
        return x + 1.0

    @udf.elementwise([dt.double], dt.double)
    def times_two(x, scope=None):
        return x * 2.0

    @udf.analytic(input_type=[dt.double], output_type=dt.double)
    def zscore(series):
        return (series - series.mean()) / series.std()


def test_udf():
    df = pd.DataFrame({'a': list('abc')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = my_string_length(t.a)

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()
    expected = t.a.execute().str.len().mul(2)
    tm.assert_series_equal(result, expected)


def test_multiple_argument_udf():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [3.0, 4.0, 5.0]})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = my_add(t.a, t.b)

    assert isinstance(expr, ir.ColumnExpr)
    assert isinstance(expr, ir.NumericColumn)
    assert isinstance(expr, ir.FloatingColumn)

    result = expr.execute()
    expected = (t.a + t.b).execute()
    tm.assert_series_equal(result, expected)


def test_multiple_argument_udf_group_by():
    df = pd.DataFrame({
        'a': [1.0, 2.0, 3.0], 'b': [3.0, 4.0, 5.0], 'key': list('aab')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = t.groupby(t.key).aggregate(my_add=my_add(t.a, t.b).sum())

    assert isinstance(expr, ir.TableExpr)
    assert isinstance(expr.my_add, ir.ColumnExpr)
    assert isinstance(expr.my_add, ir.NumericColumn)
    assert isinstance(expr.my_add, ir.FloatingColumn)

    result = expr.execute()
    expected = pd.DataFrame({
        'key': list('ab'),
        'my_add': [sum([1.0 + 3.0, 2.0 + 4.0]), sum([3.0 + 5.0])],
    })
    tm.assert_frame_equal(result, expected)


def test_udaf():
    df = pd.DataFrame({'a': list('cba')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = my_string_length_sum(t.a)

    assert isinstance(expr, ir.ScalarExpr)

    result = expr.execute()
    expected = t.a.execute().str.len().mul(2).sum()
    assert result == expected


def test_udaf_analytic():
    df = pd.DataFrame({'a': np.random.randn(4), 'key': list('abba')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = zscore(t.a)

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()

    def f(s):
        return s.sub(s.mean()).div(s.std())

    expected = f(df.a)
    tm.assert_series_equal(result, expected)


def test_udaf_analytic_group_by():
    df = pd.DataFrame({'a': np.random.randn(4), 'key': list('abba')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = zscore(t.a).over(ibis.window(group_by=t.key))

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()

    def f(s):
        return s.sub(s.mean()).div(s.std())

    expected = df.groupby('key').a.transform(f)
    tm.assert_series_equal(result, expected)


def test_udaf_groupby():
    df = pd.DataFrame({
        'a': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'b': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'key': list('ddeefff')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = t.groupby(t.key).aggregate(my_corr=my_corr(t.a, t.b))

    assert isinstance(expr, ir.TableExpr)

    result = expr.execute().sort_values('key')

    dfi = df.set_index('key')
    expected = pd.DataFrame({
        'key': list('def'),
        'my_corr': [
            dfi.loc[value, 'a'].corr(dfi.loc[value, 'b']) for value in 'def'
        ]
    })

    columns = ['key', 'my_corr']
    tm.assert_frame_equal(result[columns], expected[columns])


def test_nullable():
    t = ibis.table([('a', 'int64')])
    assert nullable(t.a.type()) == (type(None),)


def test_nullable_non_nullable_field():
    t = ibis.table([('a', dt.String(nullable=False))])
    assert nullable(t.a.type()) == ()


def test_udaf_parameter_mismatch():
    with pytest.raises(Exception):
        @udf.reduction(input_type=[dt.double], output_type=dt.double)
        def my_corr(lhs, rhs, **kwargs):
            pass


def test_udf_parameter_mismatch():
    with pytest.raises(Exception):
        @udf.reduction(input_type=[], output_type=dt.double)
        def my_corr2(lhs, **kwargs):
            pass


def test_call_multiple_udfs():
    df = pd.DataFrame({
        'a': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'b': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'key': list('ddeefff')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = times_two(add_one(t.a))
    result = expr.execute()
    expected = df.a.add(1.0).mul(2.0)
    tm.assert_series_equal(expected, result)


def test_udaf_window():
    @udf.reduction([dt.double], dt.double)
    def my_mean(series):
        return series.mean()

    df = pd.DataFrame({
        'a': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'b': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'key': list('ddeefff')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    window = ibis.trailing_window(2, order_by='a', group_by='key')
    expr = t.mutate(rolled=my_mean(t.b).over(window))
    result = expr.execute().sort_values(['key', 'a'])
    expected = df.sort_values(['key', 'a']).assign(
        rolled=lambda df: df.groupby('key').b.rolling(2).mean().reset_index(
            level=0, drop=True)
    )
    tm.assert_frame_equal(result, expected)
