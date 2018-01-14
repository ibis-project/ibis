import math
import pytest
import decimal
from pytest import param

import numpy as np
import pandas as pd

import ibis
from ibis import literal as L


@pytest.mark.parametrize(('operand_fn', 'expected_operand_fn'), [
    param(lambda t: t.float_col,
          lambda t: t.float_col,
          id='float-column-isinf'),
    param(lambda t: t.double_col,
          lambda t: t.double_col,
          id='double-column'),
    param(lambda t: ibis.literal(1.3),
          lambda t: 1.3,
          id='float-literal',
          marks=pytest.mark.xfail),  # strange fail on Postgres
    param(lambda t: ibis.literal(np.nan),
          lambda t: np.nan,
          id='nan-literal'),
    param(lambda t: ibis.literal(np.inf),
          lambda t: np.inf,
          id='inf-literal'),
    param(lambda t: ibis.literal(-np.inf),
          lambda t: -np.inf,
          id='-inf-literal')
])
@pytest.mark.parametrize(('expr_fn', 'expected_expr_fn'), [
    param(lambda o: o.isnan(), lambda o: np.isnan(o), id='isnan'),
    param(lambda o: o.isinf(), lambda o: np.isinf(o), id='isinf')
])
def test_isnan_isinf(backend, con, alltypes, df,
                     operand_fn, expected_operand_fn,
                     expr_fn, expected_expr_fn):
    expr = expr_fn(operand_fn(alltypes))
    expected = expected_expr_fn(expected_operand_fn(df))

    with backend.skip_unsupported():
        result = con.execute(expr)

    if isinstance(expected, pd.Series):
        expected = backend.default_series_rename(expected)
        backend.assert_series_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(('expr', 'expected'), [
    (L(-5).abs(), 5),
    (L(5).abs(), 5),
    (ibis.least(L(10), L(1)), 1),
    (ibis.greatest(L(10), L(1)), 10),

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
    (L(11) % 3, 11 % 3),
])
def test_math_functions_for_literals(backend, con, alltypes, df,
                                     expr, expected):
    with backend.skip_unsupported():
        result = con.execute(expr)

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == expected
