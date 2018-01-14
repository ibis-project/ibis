import pytest
import decimal

import ibis
from ibis import literal as L


@pytest.mark.parametrize(('expr', 'expected'), [
    (ibis.NA.fillna(5), 5),
    (L(5).fillna(10), 5),
    pytest.param(L(5).nullif(5), None, marks=pytest.mark.xfail),
    (L(10).nullif(5), 10),
])
def test_fillna_nullif(backend, con, expr, expected):
    with backend.skip_unsupported():
        result = con.execute(expr)

    assert result == expected


@pytest.mark.parametrize(('expr', 'expected'), [
    (ibis.coalesce(5, None, 4), 5),
    (ibis.coalesce(ibis.NA, 4, ibis.NA), 4),
    (ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14),
])
def test_coalesce(backend, con, expr, expected):
    with backend.skip_unsupported():
        result = con.execute(expr)

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == expected


def test_identical_to(backend, alltypes, con, df):
    dt = df[['tinyint_col', 'double_col']]

    expr = alltypes.tinyint_col.identical_to(alltypes.double_col)
    with backend.skip_unsupported():
        result = expr.execute()

    expected = ((dt.tinyint_col.isnull() & dt.double_col.isnull()) |
                (dt.tinyint_col == dt.double_col))

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)
