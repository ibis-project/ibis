import pandas as pd
import pandas.util.testing as tm
import pyspark
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.tests.backends import Spark

pytestmark = [pytest.mark.spark, pytest.mark.udf]

py4j = pytest.importorskip('py4j')
ps = pytest.importorskip('pyspark')
from ..udf import udf  # noqa: E402, isort:skip


@pytest.fixture(scope='session')
def con(client):
    return client


@pytest.fixture(scope='session')
def t(con):
    return con.table('udf')


@pytest.fixture(scope='session')
def df(t):
    return t.execute()


@pytest.fixture(scope='session')
def t_nan(con):
    return con.table('udf_nan')


@pytest.fixture(scope='session')
def df_nan(t_nan):
    return t_nan.execute()


@pytest.fixture(scope='session')
def t_null(con):
    return con.table('udf_null')


@pytest.fixture(scope='session')
def df_null(t_null):
    return t_null.execute()


@pytest.fixture(scope='session')
def t_random(con):
    return con.table('udf_random')


@pytest.fixture(scope='session')
def df_random(t_random):
    return t_random.execute()


@pytest.fixture(params=[[0.25, 0.75], [0.01, 0.99]])
def qs(request):
    return request.param


# elementwise UDFs


@udf.elementwise([], dt.int64)
def a_single_number(**kwargs):
    return 1


@udf.elementwise([dt.double], dt.double)
def add_one(x):
    return x + 1.0


@udf.elementwise([dt.double], dt.double)
def times_two(x, scope=None):
    return x * 2.0


@udf.elementwise(input_type=[dt.double, dt.double], output_type=dt.double)
def my_add(c1, c2, *kwargs):
    return c1 + c2


@udf.elementwise(input_type=['string'], output_type='int64')
def my_string_length(s, **kwargs):
    return len(s) * 2


# elementwise pandas UDFs


@udf.elementwise.pandas([dt.double], dt.double)
def add_one_pandas(x):
    return x + 1.0


@udf.elementwise.pandas([dt.double], dt.double)
def times_two_pandas(x, scope=None):
    return x * 2.0


@udf.elementwise.pandas([dt.double, dt.double], dt.double)
def my_add_pandas(series1, series2, *kwargs):
    return series1 + series2


@udf.elementwise.pandas(input_type=['string'], output_type='int64')
def my_string_length_pandas(series, **kwargs):
    return series.str.len() * 2


# reduction UDFs


@udf.reduction(input_type=[dt.string], output_type=dt.int64)
def my_string_length_sum(series, **kwargs):
    return (series.str.len() * 2).sum()


@udf.reduction(input_type=[dt.double, dt.double], output_type=dt.double)
def my_corr(lhs, rhs, **kwargs):
    return lhs.corr(rhs)


@udf.reduction(
    input_type=[dt.double, dt.Array(dt.double)],
    output_type=dt.Array(dt.double),
)
def quantiles(series, quantiles):
    # quantiles gets broadcasted by Spark, i.e.
    # [0.5, 0.6] -> pd.Series([[0.5, 0.6], [0.5, 0.6], ...])
    return list(series.quantile(quantiles[0]))


add_one_fns = [add_one, add_one_pandas]
times_two_fns = [times_two, times_two_pandas]
my_add_fns = [my_add, my_add_pandas]
my_string_length_fns = [my_string_length, my_string_length_pandas]


def test_spark_dtype_to_ibis_dtype():
    from ..datatypes import _SPARK_DTYPE_TO_IBIS_DTYPE

    assert len(_SPARK_DTYPE_TO_IBIS_DTYPE.keys()) == len(
        set(_SPARK_DTYPE_TO_IBIS_DTYPE.values())
    )


@pytest.mark.parametrize('fn', my_string_length_fns)
def test_udf(t, df, fn):
    expr = fn(t.a)

    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()
    expected = df.a.str.len().mul(2)
    expected = Spark.default_series_rename(expected)
    tm.assert_series_equal(result, expected)


def test_zero_argument_udf(con, t, df):
    expr = t.projection([a_single_number().name('foo')])
    result = expr.execute().foo
    expected = pd.Series([1, 1, 1], name='foo')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('fn', my_add_fns)
def test_elementwise_udf_with_non_vectors(con, fn):
    expr = fn(1.0, 2.0)
    result = con.execute(expr)
    assert result == 3.0


@pytest.mark.parametrize('fn', my_add_fns)
def test_elementwise_udf_with_non_vectors_upcast(con, fn):
    expr = fn(1, 2)
    result = con.execute(expr)
    assert result == 3.0


@pytest.mark.parametrize('fn', my_add_fns)
def test_multiple_argument_udf(con, t, df, fn):
    expr = fn(t.b, t.c)

    assert isinstance(expr, ir.ColumnExpr)
    assert isinstance(expr, ir.NumericColumn)
    assert isinstance(expr, ir.FloatingColumn)

    result = expr.execute()
    expected = df.b + df.c
    expected = Spark.default_series_rename(expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('fn', my_add_fns)
def test_multiple_argument_udf_group_by(con, t, df, fn):
    expr = t.groupby(t.key).aggregate(my_add=fn(t.b, t.c).sum()).sort_by('key')

    assert isinstance(expr, ir.TableExpr)
    assert isinstance(expr.my_add, ir.ColumnExpr)
    assert isinstance(expr.my_add, ir.NumericColumn)
    assert isinstance(expr.my_add, ir.FloatingColumn)

    result = expr.execute()
    expected = pd.DataFrame(
        {'key': list('ab'), 'my_add': [sum([1.0 + 4.0, 2.0 + 5.0]), 3.0 + 6.0]}
    )
    tm.assert_frame_equal(result, expected)


# Spark doesn't support pandas_udf GROUPED_AGG in spark.sql(). See SPARK-28422
@pytest.mark.xfail
def test_udaf(con, t, df):
    expr = my_string_length_sum(t.a)

    assert isinstance(expr, ir.ScalarExpr)

    result = expr.execute()
    expected = t.a.execute().str.len().mul(2).sum()
    expected = Spark.default_series_rename(expected)
    assert result == expected


def test_udaf_groupby(con, t_random, df_random):
    expr = (
        t_random.groupby(t_random.key)
        .aggregate(my_corr=my_corr(t_random.a, t_random.b))
        .sort_by('key')
    )

    assert isinstance(expr, ir.TableExpr)

    result = expr.execute()

    dfi = df_random.set_index('key')
    expected = pd.DataFrame(
        {
            'key': list('def'),
            'my_corr': [
                dfi.loc[value, 'a'].corr(dfi.loc[value, 'b'])
                for value in 'def'
            ],
        }
    )

    tm.assert_frame_equal(result, expected)


def test_nullable_output_not_allowed():
    d = dt.dtype('array<string>')
    d.nullable = False

    with pytest.raises(com.IbisTypeError):

        @udf.elementwise([dt.string], d)
        def str_func(x):
            pass


def test_udaf_parameter_mismatch():
    with pytest.raises(TypeError):

        @udf.reduction(input_type=[dt.double], output_type=dt.double)
        def my_corr(lhs, rhs, **kwargs):
            pass


def test_pandas_udf_zero_args_not_allowed():
    with pytest.raises(com.UnsupportedArgumentError):

        @udf.reduction(input_type=[], output_type=dt.double)
        def my_corr2(**kwargs):
            pass


@pytest.mark.parametrize('times_two_fn', times_two_fns)
@pytest.mark.parametrize('add_one_fn', add_one_fns)
def test_compose_udfs(t_random, df_random, times_two_fn, add_one_fn):
    expr = times_two_fn(add_one_fn(t_random.a))
    result = expr.execute()
    expected = df_random.a.add(1.0).mul(2.0)
    expected = Spark.default_series_rename(expected)
    tm.assert_series_equal(expected, result)


@pytest.mark.skipif(
    pyspark.__version__ < '3.0.0', reason='Requires PySpark 3.0.0 or higher'
)
def test_udaf_window(con, t_random, df_random):
    @udf.reduction(['double'], 'double')
    def my_mean(series):
        return series.mean()

    window = ibis.trailing_window(2, order_by='a', group_by='key')
    expr = t_random.mutate(rolled=my_mean(t_random.b).over(window)).sort_by(
        ['key', 'a']
    )
    result = expr.execute()
    expected = (
        df_random.sort_values(['key', 'a'])
        .assign(
            rolled=lambda df: df.groupby('key')
            .b.rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result, expected)


# Spark functions like mean() return NaN if any input is NaN, unlike pandas
# which ignores NaN values. Spark ignores Null values, not NaN
@pytest.mark.xfail(raises=AssertionError)
def test_udaf_window_nan(con, t_nan, df_nan):
    window = ibis.trailing_window(2, order_by='a', group_by='key')
    expr = t_nan.mutate(rolled=t_nan.b.mean().over(window)).sort_by(
        ['key', 'a']
    )
    result = expr.execute()
    expected = df_nan.sort_values(['key', 'a']).assign(
        rolled=lambda d: d.groupby('key')
        .b.rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    tm.assert_frame_equal(result, expected)


def test_udaf_window_null(con, t_null, df_null):
    window = ibis.trailing_window(2, order_by='a', group_by='key')
    expr = t_null.mutate(rolled=t_null.b.mean().over(window)).sort_by(
        ['key', 'a']
    )
    result = expr.execute()
    expected = df_null.sort_values(['key', 'a']).assign(
        rolled=lambda d: d.groupby('key')
        .b.rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    tm.assert_frame_equal(result, expected)


# For more info on xfail, see #2349
@pytest.mark.skipif(
    pyspark.__version__ < '3.0.0', reason='Requires PySpark 3.0.0 or higher'
)
@pytest.mark.xfail(reason='Usage of reduction UDF does not work properly')
def test_array_return_type_reduction(con, t, df, qs):
    expr = quantiles(t.b, qs)
    result = expr.execute()
    expected = df.b.quantile(qs)
    assert result == expected.tolist()


def test_array_return_type_reduction_window(con, t_random, df_random, qs):
    expr = quantiles(t_random.b, qs).over(ibis.window())
    result = expr.execute()
    expected_raw = df_random.b.quantile(qs).tolist()
    expected = pd.Series([expected_raw] * len(df_random))
    expected = Spark.default_series_rename(expected)
    tm.assert_series_equal(result, expected)
