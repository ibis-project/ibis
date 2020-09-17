import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.tests.backends import (
    Csv,
    Impala,
    MySQL,
    OmniSciDB,
    Pandas,
    Parquet,
    Postgres,
    PySpark,
    Spark,
    SQLite,
)
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
        ),
        param(
            lambda t, win: t.id.rank().over(win),
            lambda t: t.id.rank(method='min').astype('int64') - 1,
            id='rank',
        ),
        param(
            lambda t, win: t.id.dense_rank().over(win),
            lambda t: t.id.rank(method='dense').astype('int64') - 1,
            id='dense_rank',
        ),
        param(
            # these can't be equivalent, because pandas doesn't have a way to
            # compute percentile rank with a strict less-than ordering
            #
            # cume_dist() is the corresponding function in databases that
            # support window functions
            lambda t, win: t.id.percent_rank().over(win),
            lambda t: t.id.rank(pct=True),
            id='percent_rank',
            marks=pytest.mark.xpass_backends(
                [Csv, Pandas, Parquet, PySpark, OmniSciDB],
                raises=AssertionError,
            ),
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
            lambda t, win: ibis.row_number().over(win),
            lambda t: t.cumcount(),
            id='row_number',
            marks=pytest.mark.xfail_backends(
                (Pandas, Csv, Parquet),
                raises=(IndexError, com.UnboundExpressionError),
            ),
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
            # notany() over window not supported in Impala, Postgres,
            # Spark, MySQL and SQLite backends
            lambda t, win: (t.double_col == 0).notany().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id='cumnotany',
            marks=pytest.mark.xfail_backends(
                (Impala, Postgres, Spark, MySQL, SQLite)
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
            # notall() over window not supported in Impala, Postgres,
            # Spark, MySQL and SQLite backends
            lambda t, win: (t.double_col == 0).notall().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).all())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id='cumnotall',
            marks=pytest.mark.xfail_backends(
                (Impala, Postgres, Spark, MySQL, SQLite)
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
@pytest.mark.xfail_unsupported
def test_grouped_bounded_expanding_window(
    backend, alltypes, df, con, result_fn, expected_fn
):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

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
                pytest.mark.udf,
                pytest.mark.skip_backends([PySpark, Spark]),
            ],
        ),
    ],
)
# Some backends do not support non-grouped window specs
@pytest.mark.xfail_backends([OmniSciDB])
@pytest.mark.xfail_unsupported
def test_ungrouped_bounded_expanding_window(
    backend, alltypes, df, con, result_fn, expected_fn
):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

    expr = alltypes.mutate(
        val=result_fn(
            alltypes, win=ibis.window(following=0, order_by=[alltypes.id]),
        )
    )
    result = expr.execute().set_index('id').sort_index()

    column = expected_fn(df.sort_values('id'))
    expected = df.assign(val=column).set_index('id').sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.xfail_unsupported
def test_grouped_bounded_following_window(backend, alltypes, df, con):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

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
@pytest.mark.xfail_unsupported
def test_grouped_bounded_preceding_windows(
    backend, alltypes, df, con, window_fn
):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

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
            marks=pytest.mark.udf,
        ),
    ],
)
@pytest.mark.parametrize(
    ('ordered'), [param(True, id='orderered'), param(False, id='unordered')],
)
@pytest.mark.xfail_unsupported
def test_grouped_unbounded_window(
    backend, alltypes, df, con, result_fn, expected_fn, ordered
):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

    # Define a window that is
    # 1) Grouped
    # 2) Ordered if `ordered` is True
    # 3) Unbounded
    order_by = [alltypes.id] if ordered else None
    window = ibis.window(group_by=[alltypes.string_col], order_by=order_by)
    expr = alltypes.mutate(val=result_fn(alltypes, win=window,))
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
    ('result_fn', 'expected_fn'),
    [
        # Reduction ops
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            id='mean',
        ),
        param(
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            id='mean_udf',
            marks=pytest.mark.udf,
        ),
        # Analytic ops
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda df: df.float_col.shift(1),
            id='lag',
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda df: df.float_col.shift(-1),
            id='lead',
        ),
        param(
            lambda t, win: calc_zscore(t.double_col).over(win),
            lambda df: df.double_col.transform(calc_zscore.func),
            id='zscore_udf',
            marks=pytest.mark.udf,
        ),
    ],
)
@pytest.mark.parametrize(
    ('ordered'),
    [
        param(
            # Temporarily disabled on Spark and Imapala because their behavior
            # is currently inconsistent with the other backends (see #2378).
            True,
            id='orderered',
            marks=pytest.mark.skip_backends([Spark, Impala]),
        ),
        param(
            # Disabled on MySQL and PySpark because they require a defined
            # ordering for analytic ops like lag and lead.
            # Disabled on Spark because its behavior is inconsistent with other
            # backends (see #2381).
            False,
            id='unordered',
            marks=pytest.mark.skip_backends([MySQL, PySpark, Spark]),
        ),
    ],
)
# Some backends do not support non-grouped window specs
@pytest.mark.xfail_backends([OmniSciDB])
@pytest.mark.xfail_unsupported
def test_ungrouped_unbounded_window(
    backend, alltypes, df, con, result_fn, expected_fn, ordered
):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

    # Define a window that is
    # 1) Ungrouped
    # 2) Ordered if `ordered` is True
    # 3) Unbounded
    order_by = [alltypes.id] if ordered else None
    window = ibis.window(order_by=order_by)
    expr = alltypes.mutate(val=result_fn(alltypes, win=window,))
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
