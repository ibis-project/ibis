import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.udf.vectorized import analytic, reduction


@reduction(input_type=[dt.double], output_type=dt.double)
def mean_udf(s):
    return s.mean()


@analytic(input_type=[dt.double], output_type=dt.double)
def calc_zscore(s):
    return (s - s.mean()) / s.std()


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda t: t.float_col.shift(1),
            id='lag',
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda t: t.float_col.shift(-1),
            id='lead',
            marks=pytest.mark.broken(
                ["clickhouse"],
                reason="upstream is broken; returns all nulls",
            ),
        ),
        param(
            lambda t, win: t.id.rank().over(win),
            lambda t: t.id.rank(method='min').astype('int64') - 1,
            id='rank',
            marks=pytest.mark.min_server_version(clickhouse="22.8"),
        ),
        param(
            lambda t, win: t.id.dense_rank().over(win),
            lambda t: t.id.rank(method='dense').astype('int64') - 1,
            id='dense_rank',
            marks=pytest.mark.min_server_version(clickhouse="22.8"),
        ),
        param(
            lambda t, win: t.id.percent_rank().over(win),
            lambda t: t.apply(
                lambda df: (
                    df.sort_values("id")
                    .id.rank(method="min")
                    .sub(1)
                    .div(len(df) - 1)
                )
            ).reset_index(drop=True, level=[0]),
            id='percent_rank',
            marks=pytest.mark.notyet(
                ["clickhouse"],
                reason="clickhouse doesn't implement percent_rank",
            ),
        ),
        param(
            lambda t, win: t.id.cume_dist().over(win),
            lambda t: t.id.rank(method='min') / t.id.transform(len),
            id='cume_dist',
            marks=[
                pytest.mark.notimpl(["pyspark"]),
                pytest.mark.notyet(["clickhouse"]),
            ],
        ),
        param(
            lambda t, win: t.float_col.ntile(buckets=7).over(win),
            lambda t: t,
            id='ntile',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t, win: t.float_col.first().over(win),
            lambda t: t.float_col.transform('first'),
            id='first',
        ),
        param(
            lambda t, win: t.float_col.last().over(win),
            lambda t: t.float_col.transform('last'),
            id='last',
        ),
        param(
            lambda t, win: t.float_col.nth(3).over(win),
            lambda t: t.float_col.apply(
                lambda s: pd.concat(
                    [
                        pd.Series(np.nan, index=s.index[:3], dtype="float32"),
                        pd.Series(
                            s.iloc[3],
                            index=s.index[3:],
                            dtype="float32",
                        ),
                    ]
                )
            ),
            id="nth",
            marks=[
                pytest.mark.notimpl(["pandas"]),
                pytest.mark.notyet(["impala"]),
            ],
        ),
        param(
            lambda _, win: ibis.row_number().over(win),
            lambda t: t.cumcount(),
            id='row_number',
            marks=[
                pytest.mark.notimpl(["pandas"]),
                pytest.mark.min_server_version(clickhouse="22.8"),
            ],
        ),
        param(
            lambda t, win: t.double_col.cumsum().over(win),
            lambda t: t.double_col.cumsum(),
            id='cumsum',
        ),
        param(
            lambda t, win: t.double_col.cummean().over(win),
            lambda t: (
                t.double_col.expanding().mean().reset_index(drop=True, level=0)
            ),
            id='cummean',
        ),
        param(
            lambda t, win: t.float_col.cummin().over(win),
            lambda t: t.float_col.cummin(),
            id='cummin',
        ),
        param(
            lambda t, win: t.float_col.cummax().over(win),
            lambda t: t.float_col.cummax(),
            id='cummax',
        ),
        param(
            lambda t, win: (t.double_col == 0).any().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id='cumany',
        ),
        param(
            lambda t, win: (t.double_col == 0).notany().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id='cumnotany',
            marks=pytest.mark.notyet(
                (
                    "clickhouse",
                    "duckdb",
                    'impala',
                    'postgres',
                    'mysql',
                    'sqlite',
                ),
                reason="notany() over window not supported",
            ),
        ),
        param(
            lambda t, win: (t.double_col == 0).all().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: s.eq(0).all())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id='cumall',
        ),
        param(
            lambda t, win: (t.double_col == 0).notall().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).all())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id='cumnotall',
            marks=pytest.mark.notyet(
                (
                    "clickhouse",
                    "duckdb",
                    'impala',
                    'postgres',
                    'mysql',
                    'sqlite',
                ),
                reason="notall() over window not supported",
            ),
        ),
        param(
            lambda t, win: t.double_col.sum().over(win),
            lambda gb: gb.double_col.cumsum(),
            id='sum',
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda gb: (
                gb.double_col.expanding()
                .mean()
                .reset_index(drop=True, level=0)
            ),
            id='mean',
        ),
        param(
            lambda t, win: t.float_col.min().over(win),
            lambda gb: gb.float_col.cummin(),
            id='min',
        ),
        param(
            lambda t, win: t.float_col.max().over(win),
            lambda gb: gb.float_col.cummax(),
            id='max',
        ),
        param(
            lambda t, win: t.double_col.count().over(win),
            # pandas doesn't including the current row, but following=0 implies
            # that we must, so we add one to the pandas result
            lambda gb: gb.double_col.cumcount() + 1,
            id='count',
        ),
    ],
)
@pytest.mark.notimpl(["dask", "datafusion"])
def test_grouped_bounded_expanding_window(
    backend, alltypes, df, result_fn, expected_fn
):
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=ibis.window(
                following=0,
                group_by=[alltypes.string_col],
                order_by=[alltypes.id],
            ),
        )
    )

    result = expr.execute().set_index('id').sort_index()
    column = expected_fn(df.sort_values('id').groupby('string_col'))
    expected = df.assign(val=column).set_index('id').sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: (df.double_col.expanding().mean()),
            id='mean',
        ),
        param(
            # Disabled on PySpark and Spark backends becuase in pyspark<3.0.0,
            # Pandas UDFs are only supported on unbounded windows
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda df: (df.double_col.expanding().mean()),
            id='mean_udf',
            marks=[
                pytest.mark.notimpl(
                    [
                        "clickhouse",
                        "duckdb",
                        "impala",
                        "mysql",
                        "postgres",
                        "sqlite",
                    ]
                )
            ],
        ),
    ],
)
# Some backends do not support non-grouped window specs
@pytest.mark.notimpl(["dask", "datafusion"])
def test_ungrouped_bounded_expanding_window(
    backend, alltypes, df, result_fn, expected_fn
):
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=ibis.window(following=0, order_by=[alltypes.id]),
        )
    )
    result = expr.execute().set_index('id').sort_index()

    column = expected_fn(df.sort_values('id'))
    expected = df.assign(val=column).set_index('id').sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.notimpl(["dask", "datafusion", "pandas"])
def test_grouped_bounded_following_window(backend, alltypes, df):
    window = ibis.window(
        preceding=0,
        following=2,
        group_by=[alltypes.string_col],
        order_by=[alltypes.id],
    )

    expr = alltypes.mutate(val=alltypes.id.mean().over(window))

    result = expr.execute().set_index('id').sort_index()

    # shift id column before applying Pandas rolling window summarizer to
    # simulate forward looking window aggregation
    gdf = df.sort_values('id').groupby('string_col')
    gdf.id = gdf.apply(lambda t: t.id.shift(-2))
    expected = (
        df.assign(
            val=gdf.id.rolling(3, min_periods=1)
            .mean()
            .sort_index(level=1)
            .reset_index(drop=True)
        )
        .set_index('id')
        .sort_index()
    )

    # discard first 2 rows of each group to account for the shift
    n = len(gdf) * 2
    left, right = result.val.shift(-n), expected.val.shift(-n)

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    'window_fn',
    [
        param(
            lambda t: ibis.window(
                preceding=2,
                following=0,
                group_by=[t.string_col],
                order_by=[t.id],
            ),
            id='preceding-2-following-0',
        ),
        param(
            lambda t: ibis.trailing_window(
                preceding=2, group_by=[t.string_col], order_by=[t.id]
            ),
            id='trailing-2',
        ),
    ],
)
@pytest.mark.notimpl(["dask", "datafusion"])
def test_grouped_bounded_preceding_window(backend, alltypes, df, window_fn):
    window = window_fn(alltypes)

    expr = alltypes.mutate(val=alltypes.double_col.sum().over(window))

    result = expr.execute().set_index('id').sort_index()
    gdf = df.sort_values('id').groupby('string_col')
    expected = (
        df.assign(
            val=gdf.double_col.rolling(3, min_periods=1)
            .sum()
            .sort_index(level=1)
            .reset_index(drop=True)
        )
        .set_index('id')
        .sort_index()
    )

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    ('result_fn', 'expected_fn'),
    [
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda gb: (gb.double_col.transform('mean')),
            id='mean',
        ),
        param(
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda gb: (gb.double_col.transform('mean')),
            id='mean_udf',
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "dask",
                    "duckdb",
                    "impala",
                    "mysql",
                    "postgres",
                    "sqlite",
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ('ordered'),
    [
        param(
            True, id='ordered', marks=pytest.mark.notimpl(["dask", "pandas"])
        ),
        param(False, id='unordered'),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_grouped_unbounded_window(
    backend, alltypes, df, result_fn, expected_fn, ordered
):
    # Define a window that is
    # 1) Grouped
    # 2) Ordered if `ordered` is True
    # 3) Unbounded
    order_by = [alltypes.id] if ordered else None
    window = ibis.window(group_by=[alltypes.string_col], order_by=order_by)
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=window,
        )
    )
    result = expr.execute()
    result = result.set_index('id').sort_index()

    # Apply `expected_fn` onto a DataFrame that is
    # 1) Grouped
    # 2) Ordered if `ordered` is True
    df = df.sort_values('id') if ordered else df
    expected = df.assign(val=expected_fn(df.groupby('string_col')))
    expected = expected.set_index('id').sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn", "ordered"),
    [
        # Reduction ops
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            True,
            id='ordered-mean',
            marks=[
                pytest.mark.notimpl(["dask", "impala", "pandas"]),
                pytest.mark.broken(
                    ["clickhouse"], reason="upstream appears broken"
                ),
            ],
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            False,
            id='unordered-mean',
        ),
        param(
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            True,
            id='ordered-mean_udf',
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "dask",
                    "duckdb",
                    "impala",
                    "mysql",
                    "pandas",
                    "postgres",
                    "sqlite",
                ]
            ),
        ),
        param(
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            False,
            id='unordered-mean_udf',
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "duckdb",
                    "impala",
                    "mysql",
                    "postgres",
                    "sqlite",
                ]
            ),
        ),
        # Analytic ops
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda df: df.float_col.shift(1),
            True,
            id='ordered-lag',
            marks=pytest.mark.notimpl(["dask"]),
        ),
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda df: df.float_col.shift(1),
            False,
            id='unordered-lag',
            marks=pytest.mark.notimpl(["dask", "mysql", "pyspark"]),
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda df: df.float_col.shift(-1),
            True,
            id='ordered-lead',
            marks=pytest.mark.notimpl(["clickhouse", "dask"]),
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda df: df.float_col.shift(-1),
            False,
            id='unordered-lead',
            marks=pytest.mark.notimpl(
                ["clickhouse", "dask", "mysql", "pyspark"]
            ),
        ),
        param(
            lambda t, win: calc_zscore(t.double_col).over(win),
            lambda df: df.double_col.transform(calc_zscore.func),
            True,
            id='ordered-zscore_udf',
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "dask",
                    "duckdb",
                    "impala",
                    "mysql",
                    "pandas",
                    "postgres",
                    "pyspark",
                    "sqlite",
                ]
            ),
        ),
        param(
            lambda t, win: calc_zscore(t.double_col).over(win),
            lambda df: df.double_col.transform(calc_zscore.func),
            False,
            id='unordered-zscore_udf',
            marks=pytest.mark.notimpl(
                [
                    "clickhouse",
                    "duckdb",
                    "impala",
                    "mysql",
                    "postgres",
                    "pyspark",
                    "sqlite",
                ]
            ),
        ),
    ],
)
# Some backends do not support non-grouped window specs
@pytest.mark.notimpl(["datafusion"])
def test_ungrouped_unbounded_window(
    backend, alltypes, df, con, result_fn, expected_fn, ordered
):
    # Define a window that is
    # 1) Ungrouped
    # 2) Ordered if `ordered` is True
    # 3) Unbounded
    order_by = [alltypes.id] if ordered else None
    window = ibis.window(order_by=order_by)
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=window,
        )
    )
    result = expr.execute()
    result = result.set_index('id').sort_index()

    # Apply `expected_fn` onto a DataFrame that is
    # 1) Ungrouped
    # 2) Ordered if `ordered` is True
    df = df.sort_values('id') if ordered else df
    expected = df.assign(val=expected_fn(df))
    expected = expected.set_index('id').sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.notimpl(["dask", "datafusion", "impala", "pandas"])
@pytest.mark.notyet(
    ["clickhouse"],
    reason="RANGE OFFSET frame for 'DB::ColumnNullable' ORDER BY column is not implemented",  # noqa: E501
)
def test_grouped_bounded_range_window(backend, alltypes, df):
    # Explanation of the range window spec below:
    #
    # `preceding=10, following=0, order_by='id'``:
    #     The window at a particular row (call its `id` value x) will contain
    #     some other row (call its `id` value y) if x-10 <= y <= x.
    # `group_by='string_col'`:
    #     The window at a particular row will only contain other rows that
    #     have the same 'string_col' value.
    preceding = 10
    window = ibis.range_window(
        preceding=preceding,
        following=0,
        order_by='id',
        group_by='string_col',
    )
    expr = alltypes.mutate(val=alltypes.double_col.sum().over(window))
    result = expr.execute().set_index('id').sort_index()

    def gb_fn(df):
        indices = np.searchsorted(df.id, [df["prec"], df["foll"]], side="left")
        double_col = df.double_col.values
        return pd.Series(
            [double_col[start:stop].sum() for start, stop in indices.T],
            index=df.index,
        )

    res = (
        # add 1 to get the upper bound without having to make two
        # searchsorted calls
        df.assign(prec=lambda t: t.id - preceding, foll=lambda t: t.id + 1)
        .sort_values("id")
        .groupby("string_col")
        .apply(gb_fn)
        .droplevel(0)
    )
    expected = (
        df.assign(
            # Mimic our range window spec using .apply()
            val=res
        )
        .set_index('id')
        .sort_index()
    )

    backend.assert_series_equal(result.val, expected.val)


@pytest.mark.notimpl(["clickhouse", "dask", "datafusion", "pyspark"])
@pytest.mark.notyet(
    ["clickhouse"], reason="clickhouse doesn't implement percent_rank"
)
def test_percent_rank_whole_table_no_order_by(backend, alltypes, df):
    expr = alltypes.mutate(val=lambda t: t.id.percent_rank())

    result = expr.execute().set_index('id').sort_index()
    column = df.id.rank(method="min").sub(1).div(len(df) - 1)
    expected = df.assign(val=column).set_index('id').sort_index()

    backend.assert_series_equal(result.val, expected.val)
