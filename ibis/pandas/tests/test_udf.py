import pytest

import numpy as np

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.types as ir
import ibis.expr.datatypes as dt

from ibis.pandas.udf import udf, udaf, nullable
from ibis.pandas.dispatch import pause_ordering


with pause_ordering():

    @udf(input_type=[dt.string], output_type=dt.int64)
    def my_string_length(series, **kwargs):
        return series.str.len() * 2

    @udaf(input_type=[dt.string], output_type=dt.int64)
    def my_string_length_sum(series, **kwargs):
        return (series.str.len() * 2).sum()

    @udaf(input_type=[dt.double, dt.double], output_type=dt.double)
    def my_corr(lhs, rhs, **kwargs):
        return lhs.corr(rhs)


def test_udf():
    df = pd.DataFrame({'a': list('abc')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = my_string_length(t.a)

    assert isinstance(expr, ir.Expr)

    result = expr.execute()
    expected = t.a.execute().str.len().mul(2)
    tm.assert_series_equal(result, expected)


def test_udaf():
    df = pd.DataFrame({'a': list('cba')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = my_string_length_sum(t.a)

    assert isinstance(expr, ir.Expr)

    result = expr.execute()
    expected = t.a.execute().str.len().mul(2).sum()
    assert result == expected


def test_udaf_in_groupby():
    df = pd.DataFrame({
        'a': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'b': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
        'key': list('ddeefff')})
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = t.groupby(t.key).aggregate(my_corr=my_corr(t.a, t.b))

    assert isinstance(expr, ir.Expr)

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


@pytest.mark.xfail(
    raises=AssertionError, reason='Nullability is not propagated')
def test_nullable_non_nullable_field():
    t = ibis.table([('a', dt.String(nullable=False))])
    assert nullable(t.a.type()) == ()


def test_udaf_parameter_mismatch():
    with pytest.raises(TypeError):
        @udaf(input_type=[dt.double], output_type=dt.double)
        def my_corr(lhs, rhs, **kwargs):
            pass


def test_udf_parameter_mismatch():
    with pytest.raises(TypeError):
        @udf(input_type=[], output_type=dt.double)
        def my_corr2(lhs, **kwargs):
            return 1.0
