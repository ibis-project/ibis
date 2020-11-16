import collections

import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

from .. import connect
from ..udf import nullable, udf


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
def df2():
    return pd.DataFrame(
        {
            'a': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'b': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'c': np.arange(7, dtype=int).tolist(),
            'key': list('ddeefff'),
        }
    )


@pytest.fixture
def con(df, df2):
    return connect({'df': df, 'df2': df2})


@pytest.fixture
def t(con):
    return con.table('df')


@pytest.fixture
def t2(con):
    return con.table('df2')


@udf.elementwise(input_type=['string'], output_type='int64')
def my_string_length(series, **kwargs):
    return series.str.len() * 2


@udf.elementwise(input_type=[dt.double, dt.double], output_type=dt.double)
def my_add(series1, series2, **kwargs):
    return series1 + series2


@udf.reduction(['double'], 'double')
def my_mean(series):
    return series.mean()


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
def times_two(x):
    return x * 2.0


@udf.analytic(input_type=['double'], output_type='double')
def zscore(series):
    return (series - series.mean()) / series.std()


@udf.reduction(
    input_type=[dt.double], output_type=dt.Array(dt.double),
)
def quantiles(series, *, quantiles):
    return list(series.quantile(quantiles))


def test_udf(t, df):
    expr = my_string_length(t.a)

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()
    expected = df.a.str.len().mul(2)
    tm.assert_series_equal(result, expected)


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


def test_udaf_analytic_groupby(con, t, df):
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
    con = connect({'df': df})
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


def test_udf_error(t):
    @udf.elementwise(input_type=[dt.double], output_type=dt.double)
    def error_udf(s):
        raise ValueError('xxx')

    with pytest.raises(ValueError):
        error_udf(t.c).execute()


def test_compose_udfs(t2, df2):
    expr = times_two(add_one(t2.a))
    result = expr.execute()
    expected = df2.a.add(1.0).mul(2.0)
    tm.assert_series_equal(expected, result)


def test_udaf_window(t2, df2):
    window = ibis.trailing_window(2, order_by='a', group_by='key')
    expr = t2.mutate(rolled=my_mean(t2.b).over(window))
    result = expr.execute().sort_values(['key', 'a'])
    expected = df2.sort_values(['key', 'a']).assign(
        rolled=lambda df: df.groupby('key')
        .b.rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    tm.assert_frame_equal(result, expected)


def test_udaf_window_interval():
    df = pd.DataFrame(
        collections.OrderedDict(
            [
                (
                    "time",
                    pd.date_range(
                        start='20190105', end='20190101', freq='-1D'
                    ),
                ),
                ("key", [1, 2, 1, 2, 1]),
                ("value", np.arange(5)),
            ]
        )
    )

    con = connect({'df': df})
    t = con.table('df')
    window = ibis.trailing_range_window(
        ibis.interval(days=2), order_by='time', group_by='key'
    )

    expr = t.mutate(rolled=my_mean(t.value).over(window))

    result = expr.execute().sort_values(['time', 'key']).reset_index(drop=True)
    expected = (
        df.sort_values(['time', 'key'])
        .set_index('time')
        .assign(
            rolled=lambda df: df.groupby('key')
            .value.rolling('2D', closed='both')
            .mean()
            .reset_index(level=0, drop=True)
        )
    ).reset_index(drop=False)

    tm.assert_frame_equal(result, expected)


def test_multiple_argument_udaf_window():
    # PR 2035

    @udf.reduction(['double', 'double'], 'double')
    def my_wm(v, w):
        return np.average(v, weights=w)

    df = pd.DataFrame(
        {
            'a': np.arange(4, 0, dtype=float, step=-1).tolist()
            + np.random.rand(3).tolist(),
            'b': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'c': np.arange(4, dtype=float).tolist()
            + np.random.rand(3).tolist(),
            'd': np.repeat(1, 7),
            'key': list('deefefd'),
        }
    )
    con = connect({'df': df})
    t = con.table('df')
    window = ibis.trailing_window(2, order_by='a', group_by='key')
    window2 = ibis.trailing_window(1, order_by='b', group_by='key')
    expr = t.mutate(
        wm_b=my_wm(t.b, t.d).over(window),
        wm_c=my_wm(t.c, t.d).over(window),
        wm_c2=my_wm(t.c, t.d).over(window2),
    )
    result = expr.execute().sort_values(['key', 'a'])
    expected = (
        df.sort_values(['key', 'a'])
        .assign(
            wm_b=lambda df: df.groupby('key')
            .b.rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        .assign(
            wm_c=lambda df: df.groupby('key')
            .c.rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    )
    expected = expected.sort_values(['key', 'b']).assign(
        wm_c2=lambda df: df.groupby('key')
        .c.rolling(2, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    expected = expected.sort_values(['key', 'a'])

    tm.assert_frame_equal(result, expected)


def test_udaf_window_nan():
    df = pd.DataFrame(
        {
            'a': np.arange(10, dtype=float),
            'b': [3.0, np.NaN] * 5,
            'key': list('ddeefffggh'),
        }
    )
    con = connect({'df': df})
    t = con.table('df')
    window = ibis.trailing_window(2, order_by='a', group_by='key')
    expr = t.mutate(rolled=my_mean(t.b).over(window))
    result = expr.execute().sort_values(['key', 'a'])
    expected = df.sort_values(['key', 'a']).assign(
        rolled=lambda d: d.groupby('key')
        .b.rolling(3, min_periods=1)
        .apply(lambda x: x.mean(), raw=True)
        .reset_index(level=0, drop=True)
    )
    tm.assert_frame_equal(result, expected)


@pytest.fixture(params=[[0.25, 0.75], [0.01, 0.99]])
def qs(request):
    return request.param


def test_array_return_type_reduction(con, t, df, qs):
    expr = quantiles(t.b, quantiles=qs)
    result = expr.execute()
    expected = df.b.quantile(qs)
    assert result == expected.tolist()


def test_array_return_type_reduction_window(con, t, df, qs):
    expr = quantiles(t.b, quantiles=qs).over(ibis.window())
    result = expr.execute()
    expected_raw = df.b.quantile(qs).tolist()
    expected = pd.Series([expected_raw] * len(df))
    tm.assert_series_equal(result, expected)


def test_elementwise_udf_with_many_args(t2):
    @udf.elementwise(
        input_type=[dt.double] * 16 + [dt.int32] * 8, output_type=dt.double
    )
    def my_udf(
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13,
        c14,
        c15,
        c16,
        c17,
        c18,
        c19,
        c20,
        c21,
        c22,
        c23,
        c24,
    ):
        return c1

    expr = my_udf(*([t2.a] * 8 + [t2.b] * 8 + [t2.c] * 8))
    result = expr.execute()
    expected = t2.a.execute()

    tm.assert_series_equal(result, expected)
