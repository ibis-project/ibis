import numpy as np
import pandas as pd
import pytest
from pytest import mark, param

import ibis.expr.datatypes as dt
from ibis import _
from ibis import literal as L
from ibis.udf.vectorized import reduction


@reduction(input_type=[dt.double], output_type=dt.double)
def mean_udf(s):
    return s.mean()


aggregate_test_params = [
    param(
        lambda t: t.double_col.mean(),
        lambda t: t.double_col.mean(),
        id='mean',
    ),
    param(
        lambda t: mean_udf(t.double_col),
        lambda t: t.double_col.mean(),
        id='mean_udf',
        marks=[
            pytest.mark.notimpl(
                [
                    "bigquery",
                    "datafusion",
                    "postgres",
                    "clickhouse",
                    "impala",
                    "duckdb",
                    "polars",
                    "snowflake",
                    "mssql",
                    "trino",
                ]
            ),
            pytest.mark.never(["sqlite", "mysql"], reason="no udf support"),
        ],
    ),
    param(
        lambda t: t.double_col.min(),
        lambda t: t.double_col.min(),
        id='min',
    ),
    param(
        lambda t: t.double_col.max(),
        lambda t: t.double_col.max(),
        id='max',
    ),
    param(
        # int_col % 3 so there are no ties for most common value
        lambda t: (t.int_col % 3).mode(),
        lambda t: (t.int_col % 3).mode().iloc[0],
        id='mode',
        marks=pytest.mark.notyet(
            [
                "bigquery",
                "clickhouse",
                "datafusion",
                "impala",
                "mysql",
                "pyspark",
                "sqlite",
                "mssql",
                "trino",
            ]
        ),
    ),
    param(
        lambda t: (t.double_col + 5).sum(),
        lambda t: (t.double_col + 5).sum(),
        id='complex_sum',
    ),
    param(
        lambda t: t.timestamp_col.max(),
        lambda t: t.timestamp_col.max(),
        id='timestamp_max',
    ),
]

argidx_not_grouped_marks = [
    "datafusion",
    "impala",
    "mysql",
    "postgres",
    "pyspark",
    "sqlite",
    "snowflake",
    "polars",
    "mssql",
]
argidx_grouped_marks = ["dask"] + argidx_not_grouped_marks


def make_argidx_params(marks):
    marks = pytest.mark.notyet(marks)
    return [
        param(
            lambda t: t.timestamp_col.argmin(t.int_col),
            lambda s: s.timestamp_col.iloc[s.int_col.argmin()],
            id='argmin',
            marks=marks,
        ),
        param(
            lambda t: t.double_col.argmax(t.int_col),
            lambda s: s.double_col.iloc[s.int_col.argmax()],
            id='argmax',
            marks=marks,
        ),
    ]


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    aggregate_test_params + make_argidx_params(argidx_not_grouped_marks),
)
def test_aggregate(backend, alltypes, df, result_fn, expected_fn):
    expr = alltypes.aggregate(tmp=result_fn)
    result = expr.execute()

    # Create a single-row single-column dataframe with the Pandas `agg` result
    # (to match the output format of Ibis `aggregate`)
    expected = pd.DataFrame({'tmp': [expected_fn(df)]})

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    aggregate_test_params + make_argidx_params(argidx_grouped_marks),
)
def test_aggregate_grouped(backend, alltypes, df, result_fn, expected_fn):
    grouping_key_col = 'bigint_col'

    # Two (equivalent) variations:
    #  1) `group_by` then `aggregate`
    #  2) `aggregate` with `by`
    expr1 = alltypes.group_by(grouping_key_col).aggregate(tmp=result_fn)
    expr2 = alltypes.aggregate(tmp=result_fn, by=grouping_key_col)
    result1 = expr1.execute()
    result2 = expr2.execute()

    # Note: Using `reset_index` to get the grouping key as a column
    expected = (
        df.groupby(grouping_key_col).apply(expected_fn).rename('tmp').reset_index()
    )

    # Row ordering may differ depending on backend, so sort on the
    # grouping key
    result1 = result1.sort_values(by=grouping_key_col).reset_index(drop=True)
    result2 = result2.sort_values(by=grouping_key_col).reset_index(drop=True)
    expected = expected.sort_values(by=grouping_key_col).reset_index(drop=True)

    backend.assert_frame_equal(result1, expected)
    backend.assert_frame_equal(result2, expected)


@mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "postgres",
        "pyspark",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
    ]
)
def test_aggregate_multikey_group_reduction_udf(backend, alltypes, df):
    """Tests .aggregate() on a multi-key group_by with a reduction
    operation."""

    @reduction(
        input_type=[dt.double],
        output_type=dt.Struct({'mean': dt.double, 'std': dt.double}),
    )
    def mean_and_std(v):
        return v.mean(), v.std()

    grouping_key_cols = ['bigint_col', 'int_col']

    expr1 = alltypes.group_by(grouping_key_cols).aggregate(
        mean_and_std(alltypes['double_col']).destructure()
    )

    result1 = expr1.execute()

    # Note: Using `reset_index` to get the grouping key as a column
    expected = (
        df.groupby(grouping_key_cols)['double_col'].agg(['mean', 'std']).reset_index()
    )

    # Row ordering may differ depending on backend, so sort on the
    # grouping key
    result1 = result1.sort_values(by=grouping_key_cols).reset_index(drop=True)
    expected = (
        expected.sort_values(by=grouping_key_cols)
        .reset_index(drop=True)
        .assign(int_col=lambda df: df.int_col.astype("int32"))
    )

    backend.assert_frame_equal(result1, expected)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, where: t.bool_col.count(where=where),
            lambda t, where: len(t.bool_col[where].dropna()),
            id='count',
        ),
        param(
            lambda t, where: t.bool_col.nunique(where=where),
            lambda t, where: t.bool_col[where].dropna().nunique(),
            id='nunique',
            marks=pytest.mark.notimpl(["polars", "pyspark", "datafusion"]),
        ),
        param(
            lambda t, _: t.bool_col.any(),
            lambda t, _: t.bool_col.any(),
            id='any',
            marks=pytest.mark.notimpl(["polars", "datafusion"]),
        ),
        param(
            lambda t, _: t.bool_col.notany(),
            lambda t, _: ~t.bool_col.any(),
            id='notany',
            marks=pytest.mark.notimpl(["polars", "datafusion", "mssql"]),
        ),
        param(
            lambda t, _: -t.bool_col.any(),
            lambda t, _: ~t.bool_col.any(),
            id='any_negate',
            marks=pytest.mark.notimpl(["polars", "datafusion", "mssql"]),
        ),
        param(
            lambda t, _: t.bool_col.all(),
            lambda t, _: t.bool_col.all(),
            id='all',
            marks=pytest.mark.notimpl(["polars", "datafusion"]),
        ),
        param(
            lambda t, _: t.bool_col.notall(),
            lambda t, _: ~t.bool_col.all(),
            id='notall',
            marks=pytest.mark.notimpl(["polars", "datafusion", "mssql"]),
        ),
        param(
            lambda t, _: -t.bool_col.all(),
            lambda t, _: ~t.bool_col.all(),
            id='all_negate',
            marks=pytest.mark.notimpl(["polars", "datafusion", "mssql"]),
        ),
        param(
            lambda t, where: t.double_col.sum(where=where),
            lambda t, where: t.double_col[where].sum(),
            id='sum',
        ),
        param(
            lambda t, where: (t.int_col > 0).sum(where=where),
            lambda t, where: (t.int_col > 0)[where].sum(),
            id="bool_sum",
            marks=pytest.mark.notimpl(["datafusion", "pyspark", "mssql"]),
        ),
        param(
            lambda t, where: t.double_col.mean(where=where),
            lambda t, where: t.double_col[where].mean(),
            id='mean',
        ),
        param(
            lambda t, where: t.double_col.min(where=where),
            lambda t, where: t.double_col[where].min(),
            id='min',
        ),
        param(
            lambda t, where: t.double_col.max(where=where),
            lambda t, where: t.double_col[where].max(),
            id='max',
        ),
        param(
            # int_col % 3 so there are no ties for most common value
            lambda t, where: (t.int_col % 3).mode(where=where),
            lambda t, where: (t.int_col % 3)[where].mode().iloc[0],
            id='mode',
            marks=pytest.mark.notyet(
                [
                    "bigquery",
                    "clickhouse",
                    "datafusion",
                    "impala",
                    "mysql",
                    "pyspark",
                    "sqlite",
                    "mssql",
                    "trino",
                ]
            ),
        ),
        param(
            lambda t, where: t.double_col.argmin(t.int_col, where=where),
            lambda t, where: t.double_col[where].iloc[t.int_col[where].argmin()],
            id='argmin',
            marks=pytest.mark.notyet(
                [
                    "impala",
                    "mysql",
                    "postgres",
                    "pyspark",
                    "sqlite",
                    "snowflake",
                    "polars",
                    "datafusion",
                    "mssql",
                ]
            ),
        ),
        param(
            lambda t, where: t.double_col.argmax(t.int_col, where=where),
            lambda t, where: t.double_col[where].iloc[t.int_col[where].argmax()],
            id='argmax',
            marks=pytest.mark.notyet(
                [
                    "impala",
                    "mysql",
                    "postgres",
                    "pyspark",
                    "sqlite",
                    "snowflake",
                    "polars",
                    "datafusion",
                    "mssql",
                ]
            ),
        ),
        param(
            lambda t, where: t.double_col.std(how='sample', where=where),
            lambda t, where: t.double_col[where].std(ddof=1),
            id='std',
            marks=[mark.notimpl(["datafusion", "mssql"])],
        ),
        param(
            lambda t, where: t.double_col.var(how='sample', where=where),
            lambda t, where: t.double_col[where].var(ddof=1),
            id='var',
            marks=[mark.notimpl(["datafusion", "mssql"])],
        ),
        param(
            lambda t, where: t.double_col.std(how='pop', where=where),
            lambda t, where: t.double_col[where].std(ddof=0),
            id='std_pop',
            marks=[mark.notimpl(["datafusion", "mssql"])],
        ),
        param(
            lambda t, where: t.double_col.var(how='pop', where=where),
            lambda t, where: t.double_col[where].var(ddof=0),
            id='var_pop',
            marks=[mark.notimpl(["datafusion", "mssql"])],
        ),
        param(
            lambda t, where: t.string_col.approx_nunique(where=where),
            lambda t, where: t.string_col[where].nunique(),
            id='approx_nunique',
            marks=pytest.mark.notimpl(['polars', "datafusion"]),
        ),
        param(
            lambda t, where: t.double_col.arbitrary(where=where),
            lambda t, where: t.double_col[where].iloc[0],
            id='arbitrary_default',
            marks=pytest.mark.notimpl(
                [
                    'impala',
                    'postgres',
                    'mysql',
                    'sqlite',
                    'polars',
                    'datafusion',
                    "mssql",
                ]
            ),
        ),
        param(
            lambda t, where: t.double_col.arbitrary(how='first', where=where),
            lambda t, where: t.double_col[where].iloc[0],
            id='arbitrary_first',
            marks=pytest.mark.notimpl(
                [
                    'impala',
                    'postgres',
                    'mysql',
                    'sqlite',
                    'polars',
                    'datafusion',
                    "mssql",
                ]
            ),
        ),
        param(
            lambda t, where: t.double_col.arbitrary(how='last', where=where),
            lambda t, where: t.double_col[where].iloc[-1],
            id='arbitrary_last',
            marks=pytest.mark.notimpl(
                [
                    "bigquery",
                    'impala',
                    'postgres',
                    'mysql',
                    'sqlite',
                    'snowflake',
                    'polars',
                    'datafusion',
                    "mssql",
                    "trino",
                ]
            ),
        ),
        param(
            lambda t, where: t.double_col.arbitrary(how='heavy', where=where),
            lambda t, where: t.double_col[where].iloc[8],
            id='arbitrary_heavy',
            # only clickhouse implements this option
            marks=pytest.mark.notimpl(
                [
                    "bigquery",
                    "dask",
                    "datafusion",
                    "duckdb",
                    "impala",
                    "mysql",
                    "pandas",
                    "postgres",
                    "pyspark",
                    "sqlite",
                    "snowflake",
                    "polars",
                    "mssql",
                    "trino",
                ],
            ),
        ),
        param(
            lambda t, where: t.bigint_col.bit_and(where=where),
            lambda t, where: np.bitwise_and.reduce(t.bigint_col[where].values),
            id='bit_and',
            marks=[
                pytest.mark.notimpl(["dask", "polars", "datafusion", "mssql"]),
                pytest.mark.notyet(["impala", "pyspark"]),
            ],
        ),
        param(
            lambda t, where: t.bigint_col.bit_or(where=where),
            lambda t, where: np.bitwise_or.reduce(t.bigint_col[where].values),
            id='bit_or',
            marks=[
                pytest.mark.notimpl(["dask", "polars", "datafusion", "mssql"]),
                pytest.mark.notyet(["impala", "pyspark"]),
            ],
        ),
        param(
            lambda t, where: t.bigint_col.bit_xor(where=where),
            lambda t, where: np.bitwise_xor.reduce(t.bigint_col[where].values),
            id='bit_xor',
            marks=[
                pytest.mark.notimpl(["dask", "polars", "datafusion", "mssql"]),
                pytest.mark.notyet(["impala", "pyspark"]),
            ],
        ),
        param(
            lambda t, where: t.count(where=where),
            lambda t, where: len(t[where]),
            id='count_star',
        ),
        param(
            lambda t, where: t.string_col.collect(where=where),
            lambda t, where: t.string_col[where].tolist(),
            id="collect",
            marks=[
                mark.notimpl(
                    ["dask", "impala", "mysql", "sqlite", "datafusion", "mssql"]
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ('ibis_cond', 'pandas_cond'),
    [
        param(lambda _: None, lambda _: slice(None), id='no_cond'),
        param(
            lambda t: t.string_col.isin(['1', '7']),
            lambda t: t.string_col.isin(['1', '7']),
            id='is_in',
            marks=[mark.notimpl(["datafusion"])],
        ),
    ],
)
def test_reduction_ops(
    backend,
    alltypes,
    df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    expr = alltypes.agg(tmp=result_fn(alltypes, ibis_cond(alltypes))).tmp
    result = expr.execute().squeeze()
    expected = expected_fn(df, pandas_cond(df))
    try:
        np.testing.assert_allclose(result, expected, rtol=backend.reduction_tolerance)
    except TypeError:  # assert_allclose only handles numerics
        # if we're not testing numerics, then the arrays should be exactly equal
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, where: t.double_col.quantile(0.5, where=where),
            lambda t, where: t.double_col[where].quantile(0.5),
            id="quantile",
            marks=[
                mark.notimpl(
                    [
                        "bigquery",
                        "dask",
                        "datafusion",
                        "impala",
                        "mssql",
                        "mysql",
                        "polars",
                        "sqlite",
                    ]
                ),
                mark.never(
                    ["pyspark", "trino"],
                    reason="backend implements approximate quantiles",
                ),
            ],
        ),
        param(
            lambda t, where: t.double_col.quantile([0.5], where=where),
            lambda t, where: t.double_col[where].quantile([0.5]),
            id="multi-quantile",
            marks=[
                mark.notimpl(
                    [
                        "bigquery",
                        "dask",
                        "datafusion",
                        "impala",
                        "mssql",
                        "mysql",
                        "polars",
                        "sqlite",
                    ]
                ),
                mark.notyet(
                    ["snowflake"],
                    reason="backend doesn't implement array of quantiles as input",
                ),
                # strict=False untile quantilesIf works
                mark.notyet(
                    ["clickhouse"], reason="sqlglot throws a parse error", strict=False
                ),
                mark.never(
                    ["pyspark", "trino"],
                    reason="backend implements approximate quantiles",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ('ibis_cond', 'pandas_cond'),
    [
        param(lambda _: None, lambda _: slice(None), id='no_cond'),
        param(
            lambda t: t.string_col.isin(['1', '7']),
            lambda t: t.string_col.isin(['1', '7']),
            id='is_in',
            marks=[mark.notimpl(["datafusion"])],
        ),
    ],
)
def test_quantile(
    alltypes,
    df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    expr = alltypes.agg(tmp=result_fn(alltypes, ibis_cond(alltypes))).tmp
    result = expr.execute().squeeze()
    expected = expected_fn(df, pandas_cond(df))
    assert pytest.approx(result) == expected


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, where: t.G.cov(t.RBI, where=where, how="pop"),
            lambda t, where: t.G[where].cov(t.RBI[where], ddof=0),
            id='covar_pop',
            marks=[
                pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars"]),
                pytest.mark.notyet(["mysql", "impala", "sqlite"]),
            ],
        ),
        param(
            lambda t, where: t.G.cov(t.RBI, where=where, how="sample"),
            lambda t, where: t.G[where].cov(t.RBI[where], ddof=1),
            id='covar_samp',
            marks=[
                pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars"]),
                pytest.mark.notyet(["mysql", "impala", "sqlite"]),
            ],
        ),
        param(
            lambda t, where: t.G.corr(t.RBI, where=where, how="pop"),
            lambda t, where: t.G[where].corr(t.RBI[where]),
            id='corr_pop',
            marks=[
                pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars"]),
                pytest.mark.notyet(
                    ["clickhouse", "impala", "mysql", "pyspark", "sqlite"]
                ),
            ],
        ),
        param(
            lambda t, where: t.G.corr(t.RBI, where=where, how="sample"),
            lambda t, where: t.G[where].corr(t.RBI[where]),
            id='corr_samp',
            marks=[
                pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars"]),
                pytest.mark.notyet(
                    [
                        "bigquery",
                        "duckdb",
                        "impala",
                        "mysql",
                        "postgres",
                        "sqlite",
                        "snowflake",
                        "polars",
                        "trino",
                    ]
                ),
            ],
        ),
        param(
            lambda t, where: (t.G > 34.0).cov(
                t.G <= 34.0,
                where=where,
                how="pop",
            ),
            lambda t, where: (t.G[where] > 34.0).cov(t.G[where] <= 34.0, ddof=0),
            id='covar_pop_bool',
            marks=[
                pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars"]),
                pytest.mark.notyet(["mysql", "impala", "sqlite"]),
            ],
        ),
        param(
            lambda t, where: (t.G > 34.0).corr(
                t.G <= 34.0,
                where=where,
                how="pop",
            ),
            lambda t, where: (t.G[where] > 34.0).corr(t.G[where] <= 34.0),
            id='corr_pop_bool',
            marks=[
                pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars"]),
                pytest.mark.notyet(
                    ["clickhouse", "impala", "mysql", "pyspark", "sqlite"]
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ('ibis_cond', 'pandas_cond'),
    [
        param(lambda _: None, lambda _: slice(None), id='no_cond'),
        param(
            lambda t: t.yearID.isin([2009, 2015]),
            lambda t: t.yearID.isin([2009, 2015]),
            id='cond',
            marks=[
                pytest.mark.broken(
                    ["snowflake"],
                    reason=("snowflake doesn't allow quoted columns in group_by"),
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["mssql"])
def test_corr_cov(
    batting,
    batting_df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    expr = result_fn(batting, ibis_cond(batting)).name("tmp")
    result = expr.execute()

    expected = expected_fn(batting_df, pandas_cond(batting_df))

    # Backends use different algorithms for computing covariance each with
    # different amounts of numerical stability.
    #
    # This makes a generic, precise and accurate comparison function incredibly
    # fragile and tedious to write.
    assert pytest.approx(result) == expected


@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
        "snowflake",
        "mssql",
    ]
)
def test_approx_median(alltypes):
    expr = alltypes.double_col.approx_median()
    result = expr.execute()
    assert isinstance(result, float)


@mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, where, sep: (
                t.group_by('bigint_col')
                .aggregate(tmp=lambda t: t.string_col.group_concat(sep, where=where))
                .order_by('bigint_col')
            ),
            lambda t, where, sep: (
                (
                    t
                    if isinstance(where, slice)
                    else t.assign(string_col=t.string_col.where(where))
                )
                .groupby('bigint_col')
                .string_col.agg(
                    lambda s: (np.nan if pd.isna(s).all() else sep.join(s.values))
                )
                .rename('tmp')
                .sort_index()
                .reset_index()
            ),
            id='group_concat',
        )
    ],
)
@mark.parametrize(
    ("ibis_sep", "pandas_sep"),
    [
        param(":", ":", id="const"),
        param(
            L(":") + ":",
            "::",
            id="expr",
            marks=mark.notyet(["bigquery", "duckdb", "mysql", "pyspark", "trino"]),
        ),
    ],
)
@mark.parametrize(
    ('ibis_cond', 'pandas_cond'),
    [
        param(lambda _: None, lambda _: slice(None), id='no_cond'),
        param(
            lambda t: t.string_col.isin(['1', '7']),
            lambda t: t.string_col.isin(['1', '7']),
            marks=mark.notimpl(["dask"]),
            id='is_in',
        ),
        param(
            lambda t: t.string_col.notin(['1', '7']),
            lambda t: ~t.string_col.isin(['1', '7']),
            marks=mark.notimpl(["dask"]),
            id='not_in',
        ),
    ],
)
@mark.notimpl(["datafusion", "snowflake", "polars", "mssql"])
def test_group_concat(
    backend,
    alltypes,
    df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
    ibis_sep,
    pandas_sep,
):
    expr = result_fn(alltypes, ibis_cond(alltypes), ibis_sep)
    result = expr.execute()
    expected = expected_fn(df, pandas_cond(df), pandas_sep)

    backend.assert_frame_equal(result, expected)


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
@mark.notimpl(["pandas", "dask"])
def test_topk_op(alltypes, df, result_fn, expected_fn):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    t = alltypes.order_by(alltypes.string_col)
    df = df.sort_values('string_col')
    result = result_fn(t).execute()
    expected = expected_fn(df)
    assert all(result['count'].values == expected.values)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t: t.semi_join(t.string_col.topk(3), "string_col"),
            lambda t: t[
                t.string_col.isin(
                    t.groupby('string_col')['string_col'].count().head(3).index
                )
            ],
            id='string_col_filter_top3',
        )
    ],
)
@mark.notimpl(["bigquery", "datafusion", "pandas", "dask"])
def test_topk_filter_op(alltypes, df, result_fn, expected_fn):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    t = alltypes.order_by(alltypes.string_col)
    df = df.sort_values('string_col')
    expr = result_fn(t)
    result = expr.execute()
    expected = expected_fn(df)
    assert result.shape[0] == expected.shape[0]


@pytest.mark.parametrize(
    'agg_fn', [lambda s: list(s), lambda s: np.array(s)], ids=lambda obj: obj.__name__
)
@mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "postgres",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
    ]
)
def test_aggregate_list_like(backend, alltypes, df, agg_fn):
    """Tests .aggregate() where the result of an aggregation is a list-like.

    We expect the list / np.array to be treated as a scalar (in other
    words, the resulting table expression should have one element, which
    is the list / np.array).
    """

    udf = reduction(input_type=[dt.double], output_type=dt.Array(dt.double))(agg_fn)

    expr = alltypes.aggregate(result_col=udf(alltypes.double_col))
    result = expr.execute()

    # Expecting a 1-row DataFrame
    expected = pd.DataFrame({'result_col': [agg_fn(df.double_col)]})

    backend.assert_frame_equal(result, expected)


@mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "postgres",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
    ]
)
def test_aggregate_mixed_udf(backend, alltypes, df):
    """Tests .aggregate() with multiple aggregations with mixed result types.

    (In particular, one aggregation that results in an array, and other
    aggregation(s) that result in a non-array)
    """

    @reduction(input_type=[dt.double], output_type=dt.double)
    def sum_udf(v):
        return np.sum(v)

    @reduction(input_type=[dt.double], output_type=dt.Array(dt.double))
    def collect_udf(v):
        return np.array(v)

    expr = alltypes.aggregate(
        sum_col=sum_udf(alltypes.double_col),
        collect_udf=collect_udf(alltypes.double_col),
    )
    result = expr.execute()

    expected = pd.DataFrame(
        {
            'sum_col': [sum_udf.func(df.double_col)],
            'collect_udf': [collect_udf.func(df.double_col)],
        }
    )

    backend.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.notimpl(["datafusion", "pyspark"])
def test_binds_are_cast(alltypes):
    expr = alltypes.aggregate(
        high_line_count=(
            alltypes.string_col.case().when('1-URGENT', 1).else_(0).end().sum()
        )
    )

    expr.execute()


def test_agg_sort(alltypes):
    query = alltypes.aggregate(count=alltypes.count())
    query = query.order_by(alltypes.year)
    query.execute()


@pytest.mark.xfail_version(polars="==0.14.31", reason="projection of scalars is broken")
@pytest.mark.xfail_version(polars="==0.15.*", reason="projection of scalars is broken")
def test_filter(backend, alltypes, df):
    expr = (
        alltypes[_.string_col == "1"]
        .mutate(x=L(1, "int64"))
        .group_by(_.x)
        .aggregate(sum=_.double_col.sum())
    )

    # TODO: The pyspark backend doesn't apply schemas to outputs
    result = expr.execute().astype({"x": "int64"})
    expected = (
        df.loc[df.string_col == "1", :]
        .assign(x=1)
        .groupby("x")
        .double_col.sum()
        .rename("sum")
        .reset_index()
    )
    backend.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.notimpl(["polars", "datafusion", "pyspark", "mssql"])
def test_column_summary(alltypes):
    bool_col_summary = alltypes.bool_col.summary()
    expr = alltypes.aggregate(bool_col_summary)
    result = expr.execute()
    assert result.shape == (1, 7)
