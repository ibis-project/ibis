from operator import methodcaller

import numpy as np
import pandas as pd

from pandas.util import testing as tm

import pytest

import ibis
import ibis.expr.operations as ops

from ibis.pandas.dispatch import pre_execute

execute = ibis.pandas.execute

pytestmark = pytest.mark.pandas


@pytest.fixture(scope='session')
def sort_kind():
    return 'mergesort'


default = pytest.mark.parametrize('default', [ibis.NA, ibis.literal('a')])
row_offset = pytest.mark.parametrize(
    'row_offset', list(map(ibis.literal, [-1, 1, 0])))
range_offset = pytest.mark.parametrize(
    'range_offset',
    [ibis.day(), 2 * ibis.day(), -2 * ibis.day()]
)


@pytest.fixture
def row_window():
    return ibis.window(following=0, order_by='plain_int64')


@pytest.fixture
def range_window():
    return ibis.window(following=0, order_by='plain_datetimes_naive')


@default
@row_offset
def test_lead(t, df, row_offset, default, row_window):
    expr = t.dup_strings.lead(row_offset, default=default).over(row_window)
    result = expr.execute()
    expected = df.dup_strings.shift(execute(-row_offset))
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result, expected)


@default
@row_offset
def test_lag(t, df, row_offset, default, row_window):
    expr = t.dup_strings.lag(row_offset, default=default).over(row_window)
    result = expr.execute()
    expected = df.dup_strings.shift(execute(row_offset))
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result, expected)


@default
@range_offset
def test_lead_delta(t, df, range_offset, default, range_window):
    expr = t.dup_strings.lead(range_offset, default=default).over(
        range_window)
    result = expr.execute()
    expected = df[['plain_datetimes_naive', 'dup_strings']].set_index(
        'plain_datetimes_naive').squeeze().tshift(
                freq=execute(-range_offset)).reindex(
                df.plain_datetimes_naive).reset_index(drop=True)
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result, expected)


@default
@range_offset
def test_lag_delta(t, df, range_offset, default, range_window):
    expr = t.dup_strings.lag(range_offset, default=default).over(
        range_window)
    result = expr.execute()
    expected = df[['plain_datetimes_naive', 'dup_strings']].set_index(
        'plain_datetimes_naive').squeeze().tshift(
            freq=execute(range_offset)).reindex(
                df.plain_datetimes_naive).reset_index(drop=True)
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result, expected)


def test_first(t, df):
    expr = t.dup_strings.first()
    result = expr.execute()
    assert result == df.dup_strings.iloc[0]


def test_last(t, df):
    expr = t.dup_strings.last()
    result = expr.execute()
    assert result == df.dup_strings.iloc[-1]


def test_group_by_mutate_analytic(t, df):
    gb = t.groupby(t.dup_strings)
    expr = gb.mutate(
        first_value=t.plain_int64.first(),
        last_value=t.plain_strings.last(),
        avg_broadcast=t.plain_float64 - t.plain_float64.mean(),
        delta=(t.plain_int64 - t.plain_int64.lag()) / (
               t.plain_float64 - t.plain_float64.lag()),
    )
    result = expr.execute()

    gb = df.groupby('dup_strings')
    expected = df.assign(
        last_value=gb.plain_strings.transform('last'),
        first_value=gb.plain_int64.transform('first'),
        avg_broadcast=df.plain_float64 - gb.plain_float64.transform('mean'),
        delta=(
            (df.plain_int64 - gb.plain_int64.shift(1)) /
            (df.plain_float64 - gb.plain_float64.shift(1))
        )
    )

    tm.assert_frame_equal(result[expected.columns], expected)


def test_players(players, players_df):
    lagged = players.mutate(pct=lambda t: t.G - t.G.lag())
    expected = players_df.assign(
        pct=players_df.G - players_df.groupby('playerID').G.shift(1)
    )
    cols = expected.columns.tolist()
    result = lagged.execute()[cols].sort_values(cols).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_batting_filter_mean(batting, batting_df):
    expr = batting[batting.G > batting.G.mean()]
    result = expr.execute()
    expected = batting_df[batting_df.G > batting_df.G.mean()].reset_index(
        drop=True
    )
    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_zscore(players, players_df):
    expr = players.mutate(g_z=lambda t: (t.G - t.G.mean()) / t.G.std())

    gb = players_df.groupby('playerID')
    expected = players_df.assign(
        g_z=(players_df.G - gb.G.transform('mean')) / gb.G.transform('std')
    )
    cols = expected.columns.tolist()
    result = expr.execute()[cols].sort_values(cols).reset_index(
        drop=True)
    tm.assert_frame_equal(result, expected)


def test_batting_avg_change_in_games_per_year(players, players_df):
    expr = players.mutate(
        delta=lambda t: (t.G - t.G.lag()) / (t.yearID - t.yearID.lag())
    )

    gb = players_df.groupby('playerID')
    expected = players_df.assign(
        delta=(players_df.G - gb.G.shift(1)) / (
            players_df.yearID - gb.yearID.shift(1)
        )
    )

    cols = expected.columns.tolist()
    result = expr.execute()[cols].sort_values(cols).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(AssertionError, reason='NYI')
def test_batting_most_hits(players, players_df):
    expr = players.mutate(
        hits_rank=lambda t: t.H.rank().over(
            ibis.cumulative_window(order_by=ibis.desc(t.H))
        )
    )
    result = expr.execute()
    hits_rank = players_df.groupby('playerID').H.rank(
        method='min', ascending=False
    )
    expected = players_df.assign(hits_rank=hits_rank)
    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_quantile(players, players_df):
    expr = players.mutate(hits_quantile=lambda t: t.H.quantile(0.25))
    hits_quantile = players_df.groupby('playerID').H.transform(
        'quantile', 0.25
    )
    expected = players_df.assign(hits_quantile=hits_quantile)
    cols = expected.columns.tolist()
    result = expr.execute()[cols].sort_values(cols).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('op', ['sum', 'mean', 'min', 'max'])
def test_batting_specific_cumulative(batting, batting_df, op, sort_kind):
    ibis_method = methodcaller('cum{}'.format(op))
    expr = ibis_method(batting.sort_by([batting.yearID]).G)
    result = expr.execute().astype('float64')

    pandas_method = methodcaller(op)
    expected = pandas_method(
        batting_df[['G', 'yearID']].sort_values(
            'yearID', kind=sort_kind).G.expanding()
    ).reset_index(drop=True)
    tm.assert_series_equal(result, expected)


def test_batting_cumulative(batting, batting_df, sort_kind):
    expr = batting.mutate(
        more_values=lambda t: t.G.sum().over(
            ibis.cumulative_window(order_by=t.yearID)
        )
    )
    result = expr.execute()

    columns = ['G', 'yearID']
    more_values = batting_df[columns].sort_values(
        'yearID', kind=sort_kind).G.expanding().sum()
    expected = batting_df.assign(more_values=more_values)

    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_cumulative_partitioned(batting, batting_df, sort_kind):
    group_by = 'playerID'
    order_by = 'yearID'

    t = batting
    expr = t.G.sum().over(
        ibis.cumulative_window(order_by=order_by, group_by=group_by))
    expr = t.mutate(cumulative=expr)
    result = expr.execute()

    columns = [group_by, order_by, 'G']
    expected = batting_df[columns].set_index(
        order_by).groupby(group_by).G.expanding().sum().rename('cumulative')

    tm.assert_series_equal(
        result.set_index([group_by, order_by]).sort_index().cumulative,
        expected.sort_index())


def test_batting_rolling(batting, batting_df, sort_kind):
    expr = batting.mutate(
        more_values=lambda t: t.G.sum().over(
            ibis.trailing_window(5, order_by=t.yearID)
        )
    )
    result = expr.execute()

    columns = ['G', 'yearID']
    more_values = batting_df[columns].sort_values(
        'yearID', kind=sort_kind).G.rolling(5).sum()
    expected = batting_df.assign(more_values=more_values)

    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_rolling_partitioned(batting, batting_df, sort_kind):
    t = batting
    group_by = 'playerID'
    order_by = 'yearID'
    expr = t.G.sum().over(
        ibis.trailing_window(3, order_by=t[order_by], group_by=t[group_by])
    )
    expr = t.mutate(rolled=expr)
    result = expr.execute()

    columns = [group_by, order_by, 'G']
    expected = batting_df[columns].set_index(
        order_by).groupby(group_by).G.rolling(3).sum().rename('rolled')

    tm.assert_series_equal(
        result.set_index([group_by, order_by]).sort_index().rolled,
        expected.sort_index())


@pytest.mark.parametrize(
    'window',
    [
        ibis.window(order_by='yearID'),
        ibis.window(order_by='yearID', group_by='playerID'),
    ]
)
def test_window_failure_mode(batting, batting_df, window):
    # can't have order by without a following value of 0
    expr = batting.mutate(more_values=batting.G.sum().over(window))
    with pytest.raises(ibis.common.OperationNotDefinedError):
        expr.execute()


def test_scalar_broadcasting(batting, batting_df):
    expr = batting.mutate(demeaned=batting.G - batting.G.mean())
    result = expr.execute()
    expected = batting_df.assign(demeaned=batting_df.G - batting_df.G.mean())
    tm.assert_frame_equal(result, expected)


def test_mutate_with_window_after_join(sort_kind):
    left_df = pd.DataFrame({
        'ints': [0, 1, 2],
        'strings': ['a', 'b', 'c'],
        'dates': pd.date_range('20170101', periods=3),
    })
    right_df = pd.DataFrame({
        'group': [0, 1, 2] * 3,
        'value': [0, 1, np.nan, 3, 4, np.nan, 6, 7, 8],
    })
    con = ibis.pandas.connect(dict(left=left_df, right=right_df))
    left, right = map(con.table, ('left', 'right'))

    joined = left.outer_join(right, left.ints == right.group)
    proj = joined[left, right.value]
    expr = proj.groupby('ints').mutate(sum=proj.value.sum())
    result = expr.execute()
    expected = pd.DataFrame({
        'dates': pd.concat(
            [left_df.dates] * 3
        ).sort_values(kind=sort_kind).reset_index(drop=True),
        'ints': [0] * 3 + [1] * 3 + [2] * 3,
        'strings': ['a'] * 3 + ['b'] * 3 + ['c'] * 3,
        'value': [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, np.nan, np.nan, 8.0],
        'sum': [9.0] * 3 + [12.0] * 3 + [8.0] * 3,
    })
    tm.assert_frame_equal(result[expected.columns], expected)


def test_mutate_scalar_with_window_after_join():
    left_df = pd.DataFrame({'ints': range(3)})
    right_df = pd.DataFrame({
        'group': [0, 1, 2] * 3,
        'value': [0, 1, np.nan, 3, 4, np.nan, 6, 7, 8],
    })
    con = ibis.pandas.connect(dict(left=left_df, right=right_df))
    left, right = map(con.table, ('left', 'right'))

    joined = left.outer_join(right, left.ints == right.group)
    proj = joined[left, right.value]
    expr = proj.mutate(sum=proj.value.sum(), const=1)
    result = expr.execute()
    expected = pd.DataFrame({
        'ints': [0] * 3 + [1] * 3 + [2] * 3,
        'value': [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, np.nan, np.nan, 8.0],
        'sum': [29.0] * 9,
        'const': [1] * 9,
    })
    tm.assert_frame_equal(result[expected.columns], expected)


def test_project_scalar_after_join():
    left_df = pd.DataFrame({'ints': range(3)})
    right_df = pd.DataFrame({
        'group': [0, 1, 2] * 3,
        'value': [0, 1, np.nan, 3, 4, np.nan, 6, 7, 8],
    })
    con = ibis.pandas.connect(dict(left=left_df, right=right_df))
    left, right = map(con.table, ('left', 'right'))

    joined = left.outer_join(right, left.ints == right.group)
    proj = joined[left, right.value]
    expr = proj[proj.value.sum().name('sum'), ibis.literal(1).name('const')]
    result = expr.execute()
    expected = pd.DataFrame({
        'sum': [29.0] * 9,
        'const': [1] * 9,
    })
    tm.assert_frame_equal(result[expected.columns], expected)


def test_project_list_scalar():
    df = pd.DataFrame({'ints': range(3)})
    con = ibis.pandas.connect(dict(df=df))
    expr = con.table('df')
    result = expr.mutate(res=expr.ints.quantile([0.5, 0.95])).execute()
    tm.assert_series_equal(
        result.res, pd.Series([[1.0, 1.9] for _ in range(0, 3)], name='res'))


def test_window_with_preceding_expr():
    index = pd.date_range('20180101', '20180110')
    start = 2
    data = np.arange(start, start + len(index))
    df = pd.DataFrame({'value': data, 'time': index}, index=index)
    client = ibis.pandas.connect({'df': df})
    t = client.table('df')
    expected = df.set_index('time').value.rolling(
        '3d', closed='both').mean().reset_index(drop=True)
    expected.index.name = None
    day = ibis.day()
    window = ibis.trailing_window(3 * day, order_by=t.time)
    expr = t.value.mean().over(window)
    result = expr.execute()
    tm.assert_series_equal(result, expected)


def test_window_has_pre_execute_scope():
    signature = ops.Lag, ibis.pandas.PandasClient
    called = [0]

    @pre_execute.register(*signature)
    def test_pre_execute(op, client, **kwargs):
        called[0] += 1
        return {}

    data = {
        'key': list('abc'),
        'value': [1, 2, 3],
        'dup': list('ggh'),
    }
    df = pd.DataFrame(data, columns=['key', 'value', 'dup'])
    client = ibis.pandas.connect(dict(df=df))
    t = client.table('df')
    window = ibis.window(order_by='value')
    expr = t.key.lag(1).over(window).name('foo')
    result = expr.execute()
    assert result is not None

    # once in window op at the top to pickup any scope changes before computing
    # twice in window op when calling execute on the ops.Lag node at the
    # beginning of execute and once before the actual computation
    assert called[0] == 3
