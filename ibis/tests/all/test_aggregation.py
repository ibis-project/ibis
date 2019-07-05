import numpy as np
import pytest
from pytest import param

from ibis.tests.backends import Clickhouse, MySQL, SQLite


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, where: t.bool_col.count(where=where),
            lambda t, where: len(t.bool_col[where].dropna()),
            id='count',
        ),
        param(
            lambda t, where: t.bool_col.any(),
            lambda t, where: t.bool_col.any(),
            id='any',
        ),
        param(
            lambda t, where: t.bool_col.notany(),
            lambda t, where: ~t.bool_col.any(),
            id='notany',
        ),
        param(
            lambda t, where: -t.bool_col.any(),
            lambda t, where: ~t.bool_col.any(),
            id='any_negate',
        ),
        param(
            lambda t, where: t.bool_col.all(),
            lambda t, where: t.bool_col.all(),
            id='all',
        ),
        param(
            lambda t, where: t.bool_col.notall(),
            lambda t, where: ~t.bool_col.all(),
            id='notall',
        ),
        param(
            lambda t, where: -t.bool_col.all(),
            lambda t, where: ~t.bool_col.all(),
            id='all_negate',
        ),
        param(
            lambda t, where: t.double_col.sum(),
            lambda t, where: t.double_col.sum(),
            id='sum',
        ),
        param(
            lambda t, where: t.double_col.mean(),
            lambda t, where: t.double_col.mean(),
            id='mean',
        ),
        param(
            lambda t, where: t.double_col.min(),
            lambda t, where: t.double_col.min(),
            id='min',
        ),
        param(
            lambda t, where: t.double_col.max(),
            lambda t, where: t.double_col.max(),
            id='max',
        ),
        param(
            lambda t, where: t.double_col.approx_median(),
            lambda t, where: t.double_col.median(),
            id='approx_median',
            marks=pytest.mark.xpass_backends([Clickhouse]),
        ),
        param(
            lambda t, where: t.double_col.std(how='sample'),
            lambda t, where: t.double_col.std(ddof=1),
            id='std',
        ),
        param(
            lambda t, where: t.double_col.var(how='sample'),
            lambda t, where: t.double_col.var(ddof=1),
            id='var',
        ),
        param(
            lambda t, where: t.double_col.std(how='pop'),
            lambda t, where: t.double_col.std(ddof=0),
            id='std_pop',
        ),
        param(
            lambda t, where: t.double_col.var(how='pop'),
            lambda t, where: t.double_col.var(ddof=0),
            id='var_pop',
        ),
        param(
            lambda t, where: t.string_col.approx_nunique(),
            lambda t, where: t.string_col.nunique(),
            id='approx_nunique',
            marks=pytest.mark.xfail_backends([MySQL, SQLite]),
        ),
        param(
            lambda t, where: t.double_col.arbitrary(how='first'),
            lambda t, where: t.double_col.iloc[0],
            id='arbitrary_first',
        ),
        param(
            lambda t, where: t.double_col.arbitrary(how='last'),
            lambda t, where: t.double_col.iloc[-1],
            id='arbitrary_last',
        ),
    ],
)
@pytest.mark.parametrize(
    ('ibis_cond', 'pandas_cond'),
    [
        param(
            lambda t: None,
            lambda t: slice(None),
            id='no_cond',
        ),
        param(
            lambda t: t.string_col.isin(['1', '7']),
            lambda t: t.string_col.isin(['1', '7']),
            id='is_in',
        ),
    ],
)
@pytest.mark.xfail_unsupported
def test_aggregation(
    backend, alltypes, df, result_fn, expected_fn, ibis_cond, pandas_cond
):
    expr = result_fn(alltypes, ibis_cond(alltypes))
    result = expr.execute()
    expected = expected_fn(df, pandas_cond(df))
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, where: (
                t.groupby('bigint_col').aggregate(
                    tmp=lambda t: t.string_col.group_concat(',')
                )
            ),
            lambda t, where: (
                t.groupby('bigint_col')
                .string_col.agg(lambda s: ','.join(s.values))
                .rename('tmp')
                .reset_index()
            ),
            id='group_concat',
        )
    ],
)
@pytest.mark.parametrize(
    ('ibis_cond', 'pandas_cond'),
    [
        (lambda t: None, lambda t: slice(None)),
        (
            lambda t: t.string_col.isin(['1', '7']),
            lambda t: t.string_col.isin(['1', '7']),
        ),
    ],
)
@pytest.mark.xfail_unsupported
def test_group_concat(
    backend, alltypes, df, result_fn, expected_fn, ibis_cond, pandas_cond
):
    expr = result_fn(alltypes, ibis_cond(alltypes))
    result = expr.execute()
    expected = expected_fn(df, pandas_cond(df))
    assert set(result) == set(expected)
