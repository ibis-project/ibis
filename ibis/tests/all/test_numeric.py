import pytest
from pytest import param

import numpy as np
import pandas as pd

import ibis


@pytest.mark.parametrize(('operand_fn', 'expected_operand_fn'), [
    param(lambda t: t.float_col,
          lambda t: t.float_col,
          id='float-column-isinf'),
    param(lambda t: t.double_col,
          lambda t: t.double_col,
          id='double-column'),
    param(lambda t: ibis.literal(1.3),
          lambda t: 1.3,
          id='float-iteral'),
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
