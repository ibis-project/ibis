import pytest
from pytest import param

import ibis
import ibis.common as com
from ibis.tests.backends import Csv, MapD, Pandas, Parquet


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
            marks=pytest.mark.xfail_backends((MapD,)),
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
                [Csv, Pandas, Parquet], raises=AssertionError
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
def test_window(backend, alltypes, df, con, result_fn, expected_fn):
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
