import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis.expr.datatypes as dt
from ibis.tests.backends import (
    BigQuery,
    Clickhouse,
    MySQL,
    Postgres,
    PySpark,
    SQLite,
)
from ibis.udf.vectorized import reduction


@reduction(input_type=[dt.double], output_type=dt.double)
def mean_udf(s):
    return s.mean()


aggregate_test_params = [
    param(
        lambda t: t.double_col.mean(),
        lambda s: s.mean(),
        'double_col',
        id='mean',
    ),
    param(
        lambda t: mean_udf(t.double_col),
        lambda s: s.mean(),
        'double_col',
        id='mean_udf',
        marks=pytest.mark.udf,
    ),
    param(
        lambda t: t.double_col.min(),
        lambda s: s.min(),
        'double_col',
        id='min',
    ),
    param(
        lambda t: t.double_col.max(),
        lambda s: s.max(),
        'double_col',
        id='max',
    ),
    param(
        lambda t: (t.double_col + 5).sum(),
        lambda s: (s + 5).sum(),
        'double_col',
        id='complex_sum',
    ),
    param(
        lambda t: t.timestamp_col.max(),
        lambda s: s.max(),
        'timestamp_col',
        id='timestamp_max',
    ),
]


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn', 'expected_col'), aggregate_test_params,
)
@pytest.mark.xfail_unsupported
def test_aggregate(
    backend, alltypes, df, result_fn, expected_fn, expected_col
):
    expr = alltypes.aggregate(tmp=result_fn)
    result = expr.execute()

    # Create a single-row single-column dataframe with the Pandas `agg` result
    # (to match the output format of Ibis `aggregate`)
    expected = pd.DataFrame({'tmp': [df[expected_col].agg(expected_fn)]})

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn', 'expected_col'), aggregate_test_params,
)
@pytest.mark.xfail_unsupported
def test_aggregate_grouped(
    backend, alltypes, df, result_fn, expected_fn, expected_col
):
    grouping_key_col = 'bigint_col'

    # Two (equivalent) variations:
    #  1) `groupby` then `aggregate`
    #  2) `aggregate` with `by`
    expr1 = alltypes.groupby(grouping_key_col).aggregate(tmp=result_fn)
    expr2 = alltypes.aggregate(tmp=result_fn, by=grouping_key_col)
    result1 = expr1.execute()
    result2 = expr2.execute()

    # Note: Using `reset_index` to get the grouping key as a column
    expected = (
        df.groupby(grouping_key_col)[expected_col]
        .agg(expected_fn)
        .rename('tmp')
        .reset_index()
    )

    # Row ordering may differ depending on backend, so sort on the grouping key
    result1 = result1.sort_values(by=grouping_key_col).reset_index(drop=True)
    result2 = result2.sort_values(by=grouping_key_col).reset_index(drop=True)
    expected = expected.sort_values(by=grouping_key_col).reset_index(drop=True)

    pd.testing.assert_frame_equal(result1, expected)
    pd.testing.assert_frame_equal(result2, expected)


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
def test_reduction_ops(
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
# Issues #2369 #2133 #2131 #2132
@pytest.mark.xfail_backends([BigQuery, Clickhouse, MySQL, Postgres])
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
