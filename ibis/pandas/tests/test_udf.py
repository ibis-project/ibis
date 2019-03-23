import pytest

import numpy as np

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.datatypes as dt

from ibis.pandas.udf import udf, nullable


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            'a': list('abc'),
            'b': [1, 2, 3],
            'c': [4.0, 5.0, 6.0],
            'key': list('aab'),
        }
    )


@pytest.fixture
def con(df):
    return ibis.pandas.connect({'df': df})


@pytest.fixture
def t(con):
    return con.table('df')


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


@udf.elementwise([], dt.int64)
def a_single_number(**kwargs):
    return 1


@udf.reduction(
    input_type=[dt.double, dt.Array(dt.double)],
    output_type=dt.Array(dt.double),
)
def quantiles(series, quantiles):
    return list(series.quantile(quantiles))


def test_udf(t, df):
    expr = my_string_length(t.a)

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()
    expected = df.a.str.len().mul(2)
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    raises=com.UnboundExpressionError,
    reason=(
        'Need to differentiate between zero argument functions and empty scope'
    ),
)
def test_zero_argument_udf(con, t, df):
    expr = t.projection([a_single_number().name('foo')])
    result = ibis.pandas.execute(expr)
    assert result is not None


def test_elementwise_udf_with_non_vectors(con):
    expr = my_add(1.0, 2.0)
    result = con.execute(expr)
    assert result == 3.0


def test_multiple_argument_udf(con, t, df):
    expr = my_add(t.b, t.c)

    assert isinstance(expr, ir.ColumnExpr)
    assert isinstance(expr, ir.NumericColumn)
    assert isinstance(expr, ir.FloatingColumn)

    result = expr.execute()
    expected = df.b + df.c
    tm.assert_series_equal(result, expected)


def test_multiple_argument_udf_group_by(con, t, df):
    expr = t.groupby(t.key).aggregate(my_add=my_add(t.b, t.c).sum())

    assert isinstance(expr, ir.TableExpr)
    assert isinstance(expr.my_add, ir.ColumnExpr)
    assert isinstance(expr.my_add, ir.NumericColumn)
    assert isinstance(expr.my_add, ir.FloatingColumn)

    result = expr.execute()
    expected = pd.DataFrame(
        {'key': list('ab'), 'my_add': [sum([1.0 + 4.0, 2.0 + 5.0]), 3.0 + 6.0]}
    )
    tm.assert_frame_equal(result, expected)


def test_udaf(con, t, df):
    expr = my_string_length_sum(t.a)

    assert isinstance(expr, ir.ScalarExpr)

    result = expr.execute()
    expected = t.a.execute().str.len().mul(2).sum()
    assert result == expected


def test_udaf_analytic(con, t, df):
    expr = zscore(t.c)

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()

    def f(s):
        return s.sub(s.mean()).div(s.std())

    expected = f(df.c)
    tm.assert_series_equal(result, expected)


def test_udaf_analytic_group_by(con, t, df):
    expr = zscore(t.c).over(ibis.window(group_by=t.key))

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()

    def f(s):
        return s.sub(s.mean()).div(s.std())

    expected = df.groupby('key').c.transform(f)
    tm.assert_series_equal(result, expected)


def test_udaf_groupby():
    df = pd.DataFrame(
        {
            'a': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'b': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'key': list('ddeefff'),
        }
    )
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    expr = t.groupby(t.key).aggregate(my_corr=my_corr(t.a, t.b))

    assert isinstance(expr, ir.TableExpr)

    result = expr.execute().sort_values('key')

    dfi = df.set_index('key')
    expected = pd.DataFrame(
        {
            'key': list('def'),
            'my_corr': [
                dfi.loc[value, 'a'].corr(dfi.loc[value, 'b'])
                for value in 'def'
            ],
        }
    )

    columns = ['key', 'my_corr']
    tm.assert_frame_equal(result[columns], expected[columns])


def test_nullable():
    t = ibis.table([('a', 'int64')])
    assert nullable(t.a.type()) == (type(None),)


def test_nullable_non_nullable_field():
    t = ibis.table([('a', dt.String(nullable=False))])
    assert nullable(t.a.type()) == ()


def test_udaf_parameter_mismatch():
    with pytest.raises(TypeError):

        @udf.reduction(input_type=[dt.double], output_type=dt.double)
        def my_corr(lhs, rhs, **kwargs):
            pass


def test_udf_parameter_mismatch():
    with pytest.raises(TypeError):

        @udf.reduction(input_type=[], output_type=dt.double)
        def my_corr2(lhs, **kwargs):
            pass


def test_compose_udfs():
    df = pd.DataFrame(
        {
            'a': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'b': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'key': list('ddeefff'),
        }
    )
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

    df = pd.DataFrame(
        {
            'a': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'b': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'key': list('ddeefff'),
        }
    )
    con = ibis.pandas.connect({'df': df})
    t = con.table('df')
    window = ibis.trailing_window(2, order_by='a', group_by='key')
    expr = t.mutate(rolled=my_mean(t.b).over(window))
    result = expr.execute().sort_values(['key', 'a'])
    expected = df.sort_values(['key', 'a']).assign(
        rolled=lambda df: df.groupby('key')
        .b.rolling(2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    tm.assert_frame_equal(result, expected)


@pytest.fixture(params=[[0.25, 0.75], [0.01, 0.99]])
def qs(request):
    return request.param


def test_array_return_type_reduction(con, t, df, qs):
    expr = quantiles(t.b, qs)
    result = expr.execute()
    expected = df.b.quantile(qs)
    assert result == expected.tolist()


def test_array_return_type_reduction_window(con, t, df, qs):
    expr = quantiles(t.b, qs).over(ibis.window())
    result = expr.execute()
    expected_raw = df.b.quantile(qs).tolist()
    expected = pd.Series([expected_raw] * len(df))
    tm.assert_series_equal(result, expected)
