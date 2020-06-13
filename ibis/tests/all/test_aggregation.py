import numpy as np
import pytest
from pytest import param

from ibis.tests.backends import Clickhouse, MySQL, PostgreSQL, PySpark, SQLite


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
            lambda t, where: t.double_col.cov(t.float_col),
            lambda t, where: t.double_col.cov(t.float_col),
            id='covar',
        ),
        param(
            lambda t, where: t.double_col.corr(t.float_col),
            lambda t, where: t.double_col.corr(t.float_col),
            id='corr',
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
        param(lambda t: None, lambda t: slice(None), id='no_cond'),
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
            lambda t: (
                t.groupby('bigint_col').aggregate(
                    tmp=lambda t: t.string_col.group_concat(',')
                )
            ),
            lambda t: (
                t.groupby('bigint_col')
                .string_col.agg(lambda s: ','.join(s.values))
                .rename('tmp')
                .reset_index()
            ),
            id='group_concat',
        )
    ],
)
@pytest.mark.xfail_unsupported
def test_group_concat(backend, alltypes, df, result_fn, expected_fn):
    expr = result_fn(alltypes)
    result = expr.execute()
    expected = expected_fn(df)

    assert set(result.iloc[:, 1]) == set(expected.iloc[:, 1])


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t: t.string_col.topk(3),
            lambda t: t.groupby('string_col')['string_col'].count().head(3),
            id='string_col_top3',
        )
    ],
)
@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends([PySpark])  # Issue #2130
def test_topk_op(backend, alltypes, df, result_fn, expected_fn):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    t = alltypes.sort_by(alltypes.string_col)
    df = df.sort_values('string_col')
    result = result_fn(t).execute()
    expected = expected_fn(df)
    assert all(result['count'].values == expected.values)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t: t[t.string_col.topk(3)],
            lambda t: t[
                t.string_col.isin(
                    t.groupby('string_col')['string_col'].count().head(3).index
                )
            ],
            id='string_col_filter_top3',
        )
    ],
)
@pytest.mark.xfail_unsupported
# Issues #2133 #2132# #2133
@pytest.mark.xfail_backends([Clickhouse, MySQL, PostgreSQL])
@pytest.mark.skip_backends([SQLite], reason='Issue #2128')
def test_topk_filter_op(backend, alltypes, df, result_fn, expected_fn):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    t = alltypes.sort_by(alltypes.string_col)
    df = df.sort_values('string_col')
    result = result_fn(t).execute()
    expected = expected_fn(df)
    assert result.shape[0] == expected.shape[0]
