import math
import pytest
import decimal
import operator

from pytest import param

import numpy as np
import pandas as pd

import ibis
import ibis.tests.util as tu

from ibis.compat import map
from ibis import literal as L
from ibis.tests.backends import MapD


@pytest.mark.parametrize(('operand_fn', 'expected_operand_fn'), [
    param(lambda t: t.float_col, lambda t: t.float_col, id='float-column'),
    param(lambda t: t.double_col, lambda t: t.double_col, id='double-column'),
    param(lambda t: ibis.literal(1.3), lambda t: 1.3, id='float-literal'),
    param(lambda t: ibis.literal(np.nan), lambda t: np.nan, id='nan-literal'),
    param(lambda t: ibis.literal(np.inf), lambda t: np.inf, id='inf-literal'),
    param(lambda t: ibis.literal(-np.inf),
          lambda t: -np.inf,
          id='-inf-literal')
])
@pytest.mark.parametrize(('expr_fn', 'expected_expr_fn'), [
    param(operator.methodcaller('isnan'), np.isnan, id='isnan'),
    param(operator.methodcaller('isinf'), np.isinf, id='isinf')
])
@tu.skipif_unsupported
def test_isnan_isinf(backend, con, alltypes, df,
                     operand_fn, expected_operand_fn,
                     expr_fn, expected_expr_fn):
    expr = expr_fn(operand_fn(alltypes))
    expected = expected_expr_fn(expected_operand_fn(df))

    result = con.execute(expr)

    if isinstance(expected, pd.Series):
        expected = backend.default_series_rename(expected)
        backend.assert_series_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(('expr', 'expected'), [
    param(L(-5).abs(), 5, id='abs-neg'),
    param(L(5).abs(), 5, id='abs'),
    param(ibis.least(L(10), L(1)), 1, id='least'),
    param(ibis.greatest(L(10), L(1)), 10, id='greatest'),
    param(L(5.5).round(), 6.0, id='round'),
    param(L(5.556).round(2), 5.56, id='round-digits'),
    param(L(5.556).ceil(), 6.0, id='ceil'),
    param(L(5.556).floor(), 5.0, id='floor'),
    param(L(5.556).exp(), math.exp(5.556), id='expr'),
    param(L(5.556).sign(), 1, id='sign-pos'),
    param(L(-5.556).sign(), -1, id='sign-neg'),
    param(L(0).sign(), 0, id='sign-zero'),
    param(L(5.556).sqrt(), math.sqrt(5.556), id='sqrt'),
    param(L(5.556).log(2), math.log(5.556, 2), id='log-base'),
    param(L(5.556).ln(), math.log(5.556), id='ln'),
    param(L(5.556).log2(), math.log(5.556, 2), id='log2'),
    param(L(5.556).log10(), math.log10(5.556), id='log10'),
    param(L(11) % 3, 11 % 3, id='mod'),
])
@tu.skipif_backend(MapD)
@tu.skipif_unsupported
def test_math_functions_literals(backend, con, alltypes, df, expr, expected):
    result = con.execute(expr)

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ('expr_fn', 'expected_fn'),
    [
        param(
            lambda t: (-t.double_col).abs(),
            lambda t: (-t.double_col).abs(),
            id='abs-neg'
        ),
        param(
            lambda t: t.double_col.abs(),
            lambda t: t.double_col.abs(),
            id='abs'
        ),
        param(
            lambda t: t.double_col.ceil(),
            lambda t: np.ceil(t.double_col).astype('int64'),
            id='ceil'
        ),
        param(
            lambda t: t.double_col.floor(),
            lambda t: np.floor(t.double_col).astype('int64'),
            id='floor'
        ),
        param(
            lambda t: t.double_col.sign(),
            lambda t: np.sign(t.double_col),
            id='sign'
        ),
        param(
            lambda t: (-t.double_col).sign(),
            lambda t: np.sign(-t.double_col),
            id='sign-negative'
        ),
    ]
)
@tu.skipif_unsupported
def test_simple_math_functions_columns(
    backend, con, alltypes, df, expr_fn, expected_fn
):
    expr = expr_fn(alltypes)
    expected = backend.default_series_rename(expected_fn(df))
    result = con.execute(expr)
    backend.assert_series_equal(result, expected)


# we add one to double_col in this test to make sure the common case works (no
# domain errors), and we test the backends' various failure modes in each
# backend's test suite

@pytest.mark.parametrize(
    ('expr_fn', 'expected_fn'),
    [
        param(
            lambda t: t.double_col.add(1).sqrt(),
            lambda t: np.sqrt(t.double_col + 1),
            id='sqrt'
        ),
        param(
            lambda t: t.double_col.add(1).exp(),
            lambda t: np.exp(t.double_col + 1),
            id='exp'
        ),
        param(
            lambda t: t.double_col.add(1).log(2),
            lambda t: np.log2(t.double_col + 1),
            id='log2'
        ),
        param(
            lambda t: t.double_col.add(1).ln(),
            lambda t: np.log(t.double_col + 1),
            id='ln'
        ),
        param(
            lambda t: t.double_col.add(1).log10(),
            lambda t: np.log10(t.double_col + 1),
            id='log10'
        ),
    ]
)
@tu.skipif_unsupported
def test_complex_math_functions_columns(
    backend, con, alltypes, df, expr_fn, expected_fn
):
    expr = expr_fn(alltypes)
    expected = backend.default_series_rename(expected_fn(df))
    result = con.execute(expr)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('expr_fn', 'expected_fn'),
    [
        param(
            lambda be, t: t.double_col.round(),
            lambda be, t: be.round(t.double_col),
            id='round',
        ),
        param(
            lambda be, t: t.double_col.add(0.05).round(3),
            lambda be, t: be.round(t.double_col + 0.05, 3),
            id='round-with-param',
        ),
        param(
            lambda be, t: be.least(ibis.least, t.bigint_col, t.int_col),
            lambda be, t: pd.Series(list(map(min, t.bigint_col, t.int_col))),
            id='least-all-columns'
        ),
        param(
            lambda be, t: be.least(ibis.least, t.bigint_col, t.int_col, -2),
            lambda be, t: pd.Series(
                list(map(min, t.bigint_col, t.int_col, [-2] * len(t)))),
            id='least-scalar'
        ),
        param(
            lambda be, t: be.greatest(ibis.greatest, t.bigint_col, t.int_col),
            lambda be, t: pd.Series(list(map(max, t.bigint_col, t.int_col))),
            id='greatest-all-columns'
        ),
        param(
            lambda be, t: be.greatest(
                ibis.greatest, t.bigint_col, t.int_col, -2),
            lambda be, t: pd.Series(
                list(map(max, t.bigint_col, t.int_col, [-2] * len(t)))),
            id='greatest-scalar'
        ),
    ]
)
@tu.skipif_unsupported
def test_backend_specific_numerics(
    backend, con, df, alltypes, expr_fn, expected_fn
):
    expr = expr_fn(backend, alltypes)
    result = backend.default_series_rename(con.execute(expr))
    expected = backend.default_series_rename(expected_fn(backend, df))
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize('op', [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
], ids=lambda op: op.__name__)
def test_binary_arithmetic_operations(backend, alltypes, df, op):
    smallint_col = alltypes.smallint_col + 1  # make it nonzero
    smallint_series = df.smallint_col + 1

    expr = op(alltypes.double_col, smallint_col)

    result = expr.execute()
    expected = op(df.double_col, smallint_series)
    if op is operator.floordiv:
        # defined in ops.FloorDivide.output_type
        # -> returns int64 whereas pandas float64
        result = result.astype('float64')

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_exact=False,
                                check_less_precise=True)


def test_mod(backend, alltypes, df):
    expr = operator.mod(alltypes.smallint_col, alltypes.smallint_col + 1)

    result = expr.execute()
    expected = operator.mod(df.smallint_col, df.smallint_col + 1)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_dtype=False)


def test_floating_mod(backend, alltypes, df):
    if not backend.supports_floating_modulus:
        pytest.skip(
            '{} backend does not support floating modulus operation'.format(
                backend.name
            )
        )
    expr = operator.mod(alltypes.double_col, alltypes.smallint_col + 1)

    result = expr.execute()
    expected = operator.mod(df.double_col, df.smallint_col + 1)

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_exact=False,
                                check_less_precise=True)


@pytest.mark.parametrize(
    'column',
    [
        'tinyint_col',
        'smallint_col',
        'int_col',
        'bigint_col',
        'float_col',
        'double_col',
    ]
)
@pytest.mark.parametrize('denominator', [0, 0.0])
def test_divide_by_zero(backend, alltypes, df, column, denominator):
    if not backend.supports_divide_by_zero:
        pytest.skip(
            '{} does not support safe division by zero'.format(backend)
        )
    expr = alltypes[column] / denominator
    expected = backend.default_series_rename(df[column].div(denominator))
    result = expr.execute()
    backend.assert_series_equal(result, expected)
