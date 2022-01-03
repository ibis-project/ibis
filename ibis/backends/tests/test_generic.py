import decimal

import numpy as np
import pandas as pd
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.util as util
from ibis import literal as L


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (ibis.NA.fillna(5), 5),
        (L(5).fillna(10), 5),
        (L(5).nullif(5), None),
        (L(10).nullif(5), 10),
    ],
)
@pytest.mark.xfail_unsupported
def test_fillna_nullif(backend, con, expr, expected):
    if expected is None:
        # The exact kind of null value used differs per backend (and version).
        # Example 1: Pandas returns np.nan while BigQuery returns None.
        # Example 2: PySpark returns np.nan if pyspark==3.0.0, but returns None
        # if pyspark <=3.0.0.
        # TODO: Make this behavior consistent (#2365)
        assert pd.isna(con.execute(expr))
    else:
        assert con.execute(expr) == expected


@pytest.mark.only_on_backends(['pandas', 'dask', 'pyspark'])
def test_isna(backend, alltypes):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    for col in ['na_col', 'none_col']:
        result = table[table[col].isnan()].execute().reset_index(drop=True)

        expected = table_pandas[table_pandas[col].isna()].reset_index(
            drop=True
        )
        backend.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends(['pandas', 'dask', 'pyspark'])
def test_fillna(backend, alltypes):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    for col in ['na_col', 'none_col']:
        result = (
            table.mutate(filled=table[col].fillna(0.0))
            .execute()
            .reset_index(drop=True)
        )

        expected = table_pandas.assign(
            filled=table_pandas[col].fillna(0.0)
        ).reset_index(drop=True)

        backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (ibis.coalesce(5, None, 4), 5),
        (ibis.coalesce(ibis.NA, 4, ibis.NA), 4),
        (ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14),
    ],
)
@pytest.mark.xfail_unsupported
def test_coalesce(backend, con, expr, expected):
    result = con.execute(expr)

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == expected


@pytest.mark.skip_backends(['dask'])  # TODO - identicalTo - #2553
@pytest.mark.xfail_unsupported
def test_identical_to(backend, alltypes, con, sorted_df):
    sorted_alltypes = alltypes.sort_by('id')
    df = sorted_df
    dt = df[['tinyint_col', 'double_col']]

    ident = sorted_alltypes.tinyint_col.identical_to(
        sorted_alltypes.double_col
    )
    expr = sorted_alltypes['id', ident.name('tmp')].sort_by('id')
    result = expr.execute().tmp

    expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
        dt.tinyint_col == dt.double_col
    )

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('column', 'elements'),
    [
        ('int_col', [1, 2, 3]),
        ('int_col', (1, 2, 3)),
        ('string_col', ['1', '2', '3']),
        ('string_col', ('1', '2', '3')),
        ('int_col', {1}),
        ('int_col', frozenset({1})),
    ],
)
@pytest.mark.xfail_unsupported
def test_isin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.sort_by('id')
    expr = sorted_alltypes[
        'id', sorted_alltypes[column].isin(elements).name('tmp')
    ].sort_by('id')
    result = expr.execute().tmp

    expected = sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('column', 'elements'),
    [
        ('int_col', [1, 2, 3]),
        ('int_col', (1, 2, 3)),
        ('string_col', ['1', '2', '3']),
        ('string_col', ('1', '2', '3')),
        ('int_col', {1}),
        ('int_col', frozenset({1})),
    ],
)
@pytest.mark.xfail_unsupported
def test_notin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.sort_by('id')
    expr = sorted_alltypes[
        'id', sorted_alltypes[column].notin(elements).name('tmp')
    ].sort_by('id')
    result = expr.execute().tmp

    expected = ~sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('predicate_fn', 'expected_fn'),
    [
        (lambda t: t['bool_col'], lambda df: df['bool_col']),
        (lambda t: ~t['bool_col'], lambda df: ~df['bool_col']),
    ],
)
@pytest.mark.skip_backends(['dask', 'datafusion'])  # TODO - sorting - #2553
@pytest.mark.xfail_unsupported
def test_filter(backend, alltypes, sorted_df, predicate_fn, expected_fn):
    sorted_alltypes = alltypes.sort_by('id')
    table = sorted_alltypes[predicate_fn(sorted_alltypes)].sort_by('id')
    result = table.execute()
    expected = sorted_df[expected_fn(sorted_df)]

    backend.assert_frame_equal(result, expected)


@pytest.mark.only_on_backends(['dask', 'pandas', 'pyspark'])
@pytest.mark.xfail_unsupported
def test_filter_with_window_op(backend, alltypes, sorted_df):
    sorted_alltypes = alltypes.sort_by('id')
    table = sorted_alltypes
    window = ibis.window(group_by=table.id)
    table = table.filter(lambda t: t['id'].mean().over(window) > 3).sort_by(
        'id'
    )
    result = table.execute()
    expected = (
        sorted_df.groupby(['id'])
        .filter(lambda t: t['id'].mean() > 3)
        .reset_index(drop=True)
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.xfail_unsupported
def test_case_where(backend, alltypes, df):
    table = alltypes
    table = table.mutate(
        new_col=(
            ibis.case()
            .when(table['int_col'] == 1, 20)
            .when(table['int_col'] == 0, 10)
            .else_(0)
            .end()
            .cast('int64')
        )
    )

    result = table.execute()

    expected = df.copy()
    mask_0 = expected['int_col'] == 1
    mask_1 = expected['int_col'] == 0

    expected['new_col'] = 0
    expected.loc[mask_0, 'new_col'] = 20
    expected.loc[mask_1, 'new_col'] = 10
    expected['new_col'] = expected['new_col']

    backend.assert_frame_equal(result, expected)


# Pr 2635
@pytest.mark.xfail_unsupported
@pytest.mark.skip_backends(['postgres'])
def test_select_filter_mutate(backend, alltypes, df):
    """Test that select, filter and mutate are executed in right order.

    Before Pr 2635, try_fusion in analysis.py would fuse these operations
    together in a way that the order of the operations were wrong. (mutate
    was executed before filter).
    """
    t = alltypes

    # Prepare the float_col so that filter must execute
    # before the cast to get the correct result.
    t = t.mutate(
        float_col=ibis.case()
        .when(t['bool_col'], t['float_col'])
        .else_(np.nan)
        .end()
    )

    # Actual test
    t = t[t.columns]
    t = t[~t['float_col'].isnan()]
    t = t.mutate(float_col=t['float_col'].cast('int32'))
    result = t.execute()

    expected = df.copy()
    expected.loc[~df['bool_col'], 'float_col'] = None
    expected = expected[~expected['float_col'].isna()]
    expected = expected.assign(float_col=expected['float_col'].astype('int32'))

    backend.assert_frame_equal(result, expected)


def test_fillna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"value \['invalid_col'\] is not a field in.*"
    ):
        alltypes.fillna({'invalid_col': 0.0})


def test_dropna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"value 'invalid_col' is not a field in.*"
    ):
        alltypes.dropna(subset=['invalid_col'])

    with pytest.raises(ValueError, match=r".*is not in.*"):
        alltypes.dropna(how='invalid')


@pytest.mark.parametrize(
    'replacements',
    [
        0.0,
        0,
        1,
        ({'na_col': 0.0}),
        ({'na_col': 1}),
        ({'none_col': 0.0}),
        ({'none_col': 1}),
    ],
)
@pytest.mark.only_on_backends(['pandas', 'dask', 'pyspark'])
def test_fillna_table(backend, alltypes, replacements):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    result = table.fillna(replacements).execute().reset_index(drop=True)
    expected = table_pandas.fillna(replacements).reset_index(drop=True)

    # check_dtype is False here because there are dtype diffs between
    # Pyspark and Pandas on Java 8 - filling the 'none_col' with an int
    # results in float in Pyspark, and int in Pandas. This diff does
    # not exist in Java 11.
    backend.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    ('how', 'subset'),
    [
        ('any', None),
        ('any', []),
        ('any', ['int_col', 'na_col']),
        ('all', None),
        ('all', ['int_col', 'na_col']),
        ('all', 'none_col'),
    ],
)
@pytest.mark.only_on_backends(['pandas', 'dask', 'pyspark'])
def test_dropna_table(backend, alltypes, how, subset):
    table = alltypes.mutate(na_col=np.nan)
    table = table.mutate(none_col=None)
    table = table.mutate(none_col=table['none_col'].cast('float64'))
    table_pandas = table.execute()

    result = table.dropna(subset, how).execute().reset_index(drop=True)
    subset = util.promote_list(subset) if subset else table_pandas.columns
    expected = table_pandas.dropna(how=how, subset=subset).reset_index(
        drop=True
    )

    # check_dtype is False here because there are dtype diffs between
    # Pyspark and Pandas on Java 8 - the 'bool_col' of an empty DataFrame
    # is type object in Pyspark, and type bool in Pandas. This diff does
    # not exist in Java 11.
    backend.assert_frame_equal(result, expected, check_dtype=False)
