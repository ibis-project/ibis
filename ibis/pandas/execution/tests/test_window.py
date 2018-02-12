from operator import methodcaller

import numpy as np
import pandas as pd

from pandas.util import testing as tm

import pytest

import ibis

pytestmark = pytest.mark.pandas


def test_lead(t, df):
    expr = t.dup_strings.lead()
    result = expr.execute()
    expected = df.dup_strings.shift(-1)
    tm.assert_series_equal(result, expected)


def test_lag(t, df):
    expr = t.dup_strings.lag()
    result = expr.execute()
    expected = df.dup_strings.shift(1)
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
            t.plain_float64 - t.plain_float64.lag()
        )
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
    result = lagged.execute()
    expected = players_df.assign(
        pct=players_df.G - players_df.groupby('playerID').G.shift(1)
    )
    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_filter_mean(batting, batting_df):
    expr = batting[batting.G > batting.G.mean()]
    result = expr.execute()
    expected = batting_df[batting_df.G > batting_df.G.mean()].reset_index(
        drop=True
    )
    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_zscore(players, players_df):
    expr = players.mutate(g_z=lambda t: (t.G - t.G.mean()) / t.G.std())
    result = expr.execute()

    gb = players_df.groupby('playerID')
    expected = players_df.assign(
        g_z=(players_df.G - gb.G.transform('mean')) / gb.G.transform('std')
    )
    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_avg_change_in_games_per_year(players, players_df):
    expr = players.mutate(
        delta=lambda t: (t.G - t.G.lag()) / (t.yearID - t.yearID.lag())
    )
    result = expr.execute()

    gb = players_df.groupby('playerID')
    expected = players_df.assign(
        delta=(players_df.G - gb.G.shift(1)) / (
            players_df.yearID - gb.yearID.shift(1)
        )
    )

    tm.assert_frame_equal(result[expected.columns], expected)


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
    result = expr.execute()
    hits_quantile = players_df.groupby('playerID').H.transform(
        'quantile', 0.25
    )
    expected = players_df.assign(hits_quantile=hits_quantile)
    tm.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize('op', ['sum', 'mean', 'min', 'max'])
def test_batting_specific_cumulative(batting, batting_df, op):
    ibis_method = methodcaller('cum{}'.format(op))
    expr = ibis_method(batting.sort_by([batting.yearID]).G)
    result = expr.execute().astype('float64')

    pandas_method = methodcaller(op)
    expected = pandas_method(
        batting_df[['G', 'yearID']].sort_values('yearID').G.expanding()
    ).reset_index(drop=True)
    tm.assert_series_equal(result, expected)


def test_batting_cumulative(batting, batting_df):
    expr = batting.mutate(
        more_values=lambda t: t.G.sum().over(
            ibis.cumulative_window(order_by=t.yearID)
        )
    )
    result = expr.execute()

    columns = ['G', 'yearID']
    more_values = batting_df[columns].sort_values('yearID').G.cumsum()
    expected = batting_df.assign(more_values=more_values)

    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_cumulative_partitioned(batting, batting_df):
    expr = batting.mutate(
        more_values=lambda t: t.G.sum().over(
            ibis.cumulative_window(order_by=t.yearID, group_by=t.lgID)
        )
    )
    result = expr.execute().more_values

    columns = ['G', 'yearID', 'lgID']
    key = 'lgID'
    expected_result = batting_df[columns].groupby(
        key, sort=False, as_index=False
    ).apply(lambda df: df.sort_values('yearID')).groupby(
        key, sort=False
    ).G.cumsum().sort_index(level=-1)
    expected = expected_result.reset_index(
        list(range(expected_result.index.nlevels - 1)),
        drop=True
    ).reindex(batting_df.index)
    expected.name = result.name

    tm.assert_series_equal(result, expected)


def test_batting_rolling(batting, batting_df):
    expr = batting.mutate(
        more_values=lambda t: t.G.sum().over(
            ibis.trailing_window(5, order_by=t.yearID)
        )
    )
    result = expr.execute()

    columns = ['G', 'yearID']
    more_values = batting_df[columns].sort_values('yearID').G.rolling(5).sum()
    expected = batting_df.assign(more_values=more_values)

    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_rolling_partitioned(batting, batting_df):
    expr = batting.mutate(
        more_values=lambda t: t.G.sum().over(
            ibis.trailing_window(3, order_by=t.yearID, group_by=t.lgID)
        )
    )
    result = expr.execute().more_values

    columns = ['G', 'yearID', 'lgID']
    key = 'lgID'
    expected_result = batting_df[columns].groupby(
        key, sort=False, as_index=False
    ).apply(lambda df: df.sort_values('yearID')).groupby(
        key, sort=False
    ).G.rolling(3).sum().sort_index(level=-1)
    expected = expected_result.reset_index(
        list(range(expected_result.index.nlevels - 1)),
        drop=True
    ).reindex(batting_df.index)
    expected.name = result.name

    tm.assert_series_equal(result, expected)


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


def test_mutate_with_window_after_join():
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
        ).sort_values().reset_index(drop=True),
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
