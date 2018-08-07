import pytest
import decimal

import ibis

from ibis import literal as L

import ibis.tests.util as tu

from ibis.tests.backends import MapD


@pytest.mark.parametrize(('expr', 'expected'), [
    (ibis.NA.fillna(5), 5),
    (L(5).fillna(10), 5),
    pytest.param(L(5).nullif(5), None, marks=pytest.mark.xfail),
    (L(10).nullif(5), 10),
])
@tu.skipif_unsupported
def test_fillna_nullif(backend, con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(('expr', 'expected'), [
    (ibis.coalesce(5, None, 4), 5),
    (ibis.coalesce(ibis.NA, 4, ibis.NA), 4),
    (ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14),
])
@tu.skipif_unsupported
def test_coalesce(backend, con, expr, expected):
    result = con.execute(expr)

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == expected


@tu.skipif_unsupported
@tu.skipif_backend(MapD)
def test_identical_to(backend, sorted_alltypes, con, sorted_df):
    df = sorted_df
    dt = df[['tinyint_col', 'double_col']]

    ident = sorted_alltypes.tinyint_col.identical_to(
        sorted_alltypes.double_col)
    expr = sorted_alltypes['id', ident.name('tmp')].sort_by('id')
    result = expr.execute().tmp

    expected = ((dt.tinyint_col.isnull() & dt.double_col.isnull()) |
                (dt.tinyint_col == dt.double_col))

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(('column', 'elements'), [
    ('int_col', [1, 2, 3]),
    ('int_col', (1, 2, 3)),
    ('string_col', ['1', '2', '3']),
    ('string_col', ('1', '2', '3')),
    pytest.mark.xfail(('int_col', {1}), raises=TypeError,
                      reason='Not yet implemented'),
    pytest.mark.xfail(('int_col', frozenset({1})), raises=TypeError,
                      reason='Not yet implemented'),
])
@tu.skipif_unsupported
@tu.skipif_backend(MapD)
def test_isin(backend, sorted_alltypes, sorted_df, column, elements):
    expr = sorted_alltypes[
        'id', sorted_alltypes[column].isin(elements).name('tmp')
    ].sort_by('id')
    result = expr.execute().tmp

    expected = sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(('column', 'elements'), [
    ('int_col', [1, 2, 3]),
    ('int_col', (1, 2, 3)),
    ('string_col', ['1', '2', '3']),
    ('string_col', ('1', '2', '3')),
    pytest.mark.xfail(('int_col', {1}), raises=TypeError,
                      reason='Not yet implemented'),
    pytest.mark.xfail(('int_col', frozenset({1})), raises=TypeError,
                      reason='Not yet implemented'),
])
@tu.skipif_unsupported
@tu.skipif_backend(MapD)
def test_notin(backend, sorted_alltypes, sorted_df, column, elements):
    expr = sorted_alltypes[
        'id',
        sorted_alltypes[column].notin(elements).name('tmp')
    ].sort_by('id')
    result = expr.execute().tmp

    expected = ~sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)
