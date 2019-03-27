import pytest

import math
import decimal
import operator
import functools
from operator import methodcaller

import numpy as np
import pandas as pd
import pandas.util.testing as tm  # noqa: E402
import ibis.expr.datatypes as dt  # noqa: E402


pytestmark = pytest.mark.pandas


@pytest.mark.parametrize(
    'op',
    [
        # comparison
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ]
)
def test_binary_operations(t, df, op):
    expr = op(t.plain_float64, t.plain_int64)
    result = expr.execute()
    expected = op(df.plain_float64, df.plain_int64)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('op', [operator.and_, operator.or_, operator.xor])
def test_binary_boolean_operations(t, df, op):
    expr = op(t.plain_int64 == 1, t.plain_int64 == 2)
    result = expr.execute()
    expected = op(df.plain_int64 == 1, df.plain_int64 == 2)
    tm.assert_series_equal(result, expected)


def operate(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except decimal.InvalidOperation:
            return decimal.Decimal('NaN')
    return wrapper


@pytest.mark.parametrize(
    ('ibis_func', 'pandas_func'),
    [
        (methodcaller('round'), lambda x: np.int64(round(x))),
        (
            methodcaller('round', 2),
            lambda x: x.quantize(decimal.Decimal('.00'))
        ),
        (
            methodcaller('round', 0),
            lambda x: x.quantize(decimal.Decimal('0.'))
        ),
        (methodcaller('ceil'), lambda x: decimal.Decimal(math.ceil(x))),
        (methodcaller('floor'), lambda x: decimal.Decimal(math.floor(x))),
        (methodcaller('exp'), methodcaller('exp')),
        (
            methodcaller('sign'),
            lambda x: x if not x else decimal.Decimal(1).copy_sign(x)
        ),
        (methodcaller('sqrt'), operate(lambda x: x.sqrt())),
        (
            methodcaller('log', 2),
            operate(lambda x: x.ln() / decimal.Decimal(2).ln())
        ),
        (methodcaller('ln'), operate(lambda x: x.ln())),
        (
            methodcaller('log2'),
            operate(lambda x: x.ln() / decimal.Decimal(2).ln())
        ),
        (methodcaller('log10'), operate(lambda x: x.log10())),
    ]
)
def test_math_functions_decimal(t, df, ibis_func, pandas_func):
    dtype = dt.Decimal(12, 3)
    result = ibis_func(t.float64_as_strings.cast(dtype)).execute()
    context = decimal.Context(prec=dtype.precision)
    expected = df.float64_as_strings.apply(
        lambda x: context.create_decimal(x).quantize(
            decimal.Decimal(
                '{}.{}'.format(
                    '0' * (dtype.precision - dtype.scale),
                    '0' * dtype.scale
                )
            )
        )
    ).apply(pandas_func)

    result[result.apply(math.isnan)] = -99999
    expected[expected.apply(math.isnan)] = -99999
    tm.assert_series_equal(result, expected)


def test_round_decimal_with_negative_places(t, df):
    type = dt.Decimal(12, 3)
    expr = t.float64_as_strings.cast(type).round(-1)
    result = expr.execute()
    expected = pd.Series(
        list(map(decimal.Decimal, ['1.0E+2', '2.3E+2', '-1.00E+3'])),
        name='float64_as_strings'
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('ibis_func', 'pandas_func'),
    [
        (lambda x: x.clip(lower=0), lambda x: x.clip(lower=0)),
        (lambda x: x.clip(lower=0.0), lambda x: x.clip(lower=0.0)),
        (lambda x: x.clip(upper=0), lambda x: x.clip(upper=0)),
        (lambda x: x.clip(lower=x - 1, upper=x + 1),
         lambda x: x.clip(lower=x - 1, upper=x + 1)),
        (lambda x: x.clip(lower=0, upper=1),
         lambda x: x.clip(lower=0, upper=1)),
        (lambda x: x.clip(lower=0, upper=1.0),
         lambda x: x.clip(lower=0, upper=1.0)),
    ]
)
def test_clip(t, df, ibis_func, pandas_func):
    result = ibis_func(t.float64_with_zeros).execute()
    expected = pandas_func(df.float64_with_zeros)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('ibis_func', 'pandas_func'),
    [
        (lambda x: x.quantile([0.25, 0.75]),
         lambda x: list(x.quantile([0.25, 0.75]))),
    ]
)
@pytest.mark.parametrize('column', ['float64_with_zeros', 'int64_with_zeros'])
def test_quantile_list(t, df, ibis_func, pandas_func, column):
    expr = ibis_func(t[column])
    result = expr.execute()
    expected = pandas_func(df[column])
    assert result == expected


@pytest.mark.parametrize(
    ('ibis_func', 'pandas_func'),
    [
        (lambda x: x.quantile(0),
         lambda x: x.quantile(0)),
        (lambda x: x.quantile(1),
         lambda x: x.quantile(1)),
        (lambda x: x.quantile(0.5, interpolation='linear'),
         lambda x: x.quantile(0.5, interpolation='linear')),
    ]
)
def test_quantile_scalar(t, df, ibis_func, pandas_func):
    result = ibis_func(t.float64_with_zeros).execute()
    expected = pandas_func(df.float64_with_zeros)

    result = ibis_func(t.int64_with_zeros).execute()
    expected = pandas_func(df.int64_with_zeros)
    assert result == expected


@pytest.mark.parametrize(
    ('ibis_func', 'exc'),
    [
        # no lower/upper specified
        (lambda x: x.clip(), ValueError),

        # out of range on quantile
        (lambda x: x.quantile(5.0), ValueError),

        # invalid interpolation arg
        (lambda x: x.quantile(0.5, interpolation='foo'), ValueError),
    ]
)
def test_arraylike_functions_transform_errors(t, df, ibis_func, exc):
    with pytest.raises(exc):
        ibis_func(t.float64_with_zeros).execute()


def test_quantile_array_access(client, t, df):
    quantile = t.float64_with_zeros.quantile([0.25, 0.5])
    expr = quantile[0], quantile[1]
    result = tuple(map(client.execute, expr))
    expected = tuple(df.float64_with_zeros.quantile([0.25, 0.5]))
    assert result == expected
