import ibis
import pytest
import numpy as np
import pandas as pd

from pytest import param
import ibis.tests.util as tu


@pytest.mark.parametrize(('result_fn', 'expected_fn'), [
    param(
        lambda t: t.float_col.lag(),
        lambda t: t.float_col.shift(1),
        id='lag'
    ),
    param(
        lambda t: t.float_col.lead(),
        lambda t: t.float_col.shift(-1),
        id='lead'
    ),
    param(
        lambda t: t.float_col.rank(),
        lambda t: t,
        id='rank',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.float_col.dense_rank(),
        lambda t: t,
        id='dense_rank',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.float_col.percent_rank(),
        lambda t: t,
        id='percent_rank',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.float_col.ntile(buckets=7),
        lambda t: t,
        id='ntile',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.float_col.first(),
        lambda t: t.float_col.head(1),
        id='first',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.float_col.last(),
        lambda t: t.float_col.tail(1),
        id='last',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.float_col.first().over(ibis.window(preceding=10)),
        lambda t: t,
        id='first_preceding',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.float_col.first().over(ibis.window(following=10)),
        lambda t: t,
        id='first_following',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: ibis.row_number(),
        lambda t: pd.Series(np.arange(len(t))),
        id='row_number',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.double_col.cumsum(),
        lambda t: t.double_col.cumsum(),
        id='cumsum'
    ),
    param(
        lambda t: t.double_col.cummean(),
        lambda t: t.double_col.expanding().mean().reset_index(
            drop=True, level=0
        ),
        id='cummean'
    ),
    param(
        lambda t: t.float_col.cummin(),
        lambda t: t.float_col.cummin(),
        id='cummin'
    ),
    param(
        lambda t: t.float_col.cummax(),
        lambda t: t.float_col.cummax(),
        id='cummax'
    ),
    param(
        lambda t: (t.double_col == 0).cumany(),
        lambda t: t.double_col.expanding().agg(
            lambda s: (s == 0).any()
        ).reset_index(drop=True, level=0).astype(bool),
        id='cumany'
    ),
    param(
        lambda t: (t.double_col == 0).cumall(),
        lambda t: t.double_col.expanding().agg(
            lambda s: (s == 0).all()
        ).reset_index(drop=True, level=0).astype(bool),
        id='cumall'
    ),
    param(
        lambda t: t.double_col.sum(),
        lambda gb: gb.double_col.transform('sum'),
        id='sum',
        marks=pytest.mark.xfail,
    ),
    param(
        lambda t: t.double_col.mean(),
        lambda gb: gb.double_col.transform('mean'),
        id='mean'
    ),
    param(
        lambda t: t.float_col.min(),
        lambda gb: gb.float_col.transform('min'),
        id='min'
    ),
    param(
        lambda t: t.float_col.max(),
        lambda gb: gb.float_col.transform('max'),
        id='max'
    ),
])
@tu.skipif_unsupported
def test_window(backend, analytic_alltypes, df, con, result_fn, expected_fn):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )

    expr = analytic_alltypes.mutate(value=result_fn)

    column = expected_fn(df.sort_values('id').groupby('string_col'))
    expected = (df.assign(value=column)
                  .set_index('id')
                  .sort_index())

    result = expr.execute().set_index('id').sort_index()

    left, right = result.value, expected.value
    backend.assert_series_equal(left, right)
