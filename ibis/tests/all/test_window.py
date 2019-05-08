import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.tests.util as tu


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
            lambda t, win: t.float_col.rank().over(win),
            lambda t: t.float_col.rank(method='min'),
            id='rank',
        ),
        param(
            lambda t, win: t.float_col.dense_rank().over(win),
            lambda t: t.float_col.rank(method='dense'),
            id='dense_rank',
        ),
        param(
            # these can't be equivalent, because pandas doesn't have a way to
            # compute percentile rank with a strict less-than ordering
            #
            # cume_dist() is the corresponding function in databases that
            # support window functions
            lambda t, win: t.float_col.percent_rank().over(win),
            lambda t: t.float_col.rank(pct=True),
            id='percent_rank',
            marks=pytest.mark.xfail(raises=AssertionError),
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
            lambda t, win: t.float_col.first().over(ibis.window(preceding=10)),
            lambda t: t,
            id='first_preceding',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t, win: t.float_col.first().over(ibis.window(following=10)),
            lambda t: t,
            id='first_following',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t, win: ibis.row_number().over(win),
            lambda t: pd.Series(np.arange(len(t))),
            id='row_number',
            marks=pytest.mark.xfail(raises=AttributeError),
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
            lambda t, win: (t.double_col == 0).cumany().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id='cumany',
        ),
        param(
            lambda t, win: (t.double_col == 0).cumall().over(win),
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
    ],
)
@tu.skipif_unsupported
def test_window(backend, analytic_alltypes, df, con, result_fn, expected_fn):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

    expr = analytic_alltypes.mutate(
        value=result_fn(
            analytic_alltypes,
            win=ibis.window(
                following=0, group_by=["string_col"], order_by=["id"]
            ),
        )
    )

    column = expected_fn(df.sort_values('id').groupby('string_col'))
    expected = df.assign(value=column).set_index('id').sort_index()

    result = expr.execute().set_index('id').sort_index()

    left, right = result.value, expected.value
    backend.assert_series_equal(left, right)
