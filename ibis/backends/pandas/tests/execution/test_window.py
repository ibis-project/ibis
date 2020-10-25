from operator import methodcaller

import numpy as np
import pandas as pd
import pytest
from pandas.util import testing as tm

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.scope import Scope
from ibis.expr.window import get_preceding_value, rows_with_max_lookback
from ibis.udf.vectorized import reduction

from ... import PandasClient, connect, execute
from ...aggcontext import AggregationContext, window_agg_udf
from ...dispatch import pre_execute
from ...execution.window import get_aggcontext

pytestmark = pytest.mark.pandas


# These custom classes are used inn test_custom_window_udf


class CustomInterval:
    def __init__(self, value):
        self.value = value

    # These are necessary because ibis.expr.window
    # will compare preceding and following
    # with 0 to see if they are valid
    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other


class CustomWindow(ibis.expr.window.Window):
    """ This is a dummy custom window that return n preceding rows
    where n is defined by CustomInterval.value."""

    def _replace(self, **kwds):
        new_kwds = dict(
            group_by=kwds.get('group_by', self._group_by),
            order_by=kwds.get('order_by', self._order_by),
            preceding=kwds.get('preceding', self.preceding),
            following=kwds.get('following', self.following),
            max_lookback=kwds.get('max_lookback', self.max_lookback),
            how=kwds.get('how', self.how),
        )
        return CustomWindow(**new_kwds)


class CustomAggContext(AggregationContext):
    def __init__(
        self, parent, group_by, order_by, output_type, max_lookback, preceding
    ):
        super().__init__(
            parent=parent,
            group_by=group_by,
            order_by=order_by,
            output_type=output_type,
            max_lookback=max_lookback,
        )
        self.preceding = preceding

    def agg(self, grouped_data, function, *args, **kwargs):
        upper_indices = pd.Series(range(1, len(self.parent) + 2))
        window_sizes = (
            grouped_data.rolling(self.preceding.value + 1)
            .count()
            .reset_index(drop=True)
        )
        lower_indices = upper_indices - window_sizes
        mask = upper_indices.notna()

        result_index = grouped_data.obj.index

        result = window_agg_udf(
            grouped_data,
            function,
            lower_indices,
            upper_indices,
            mask,
            result_index,
            self.dtype,
            self.max_lookback,
            *args,
            **kwargs,
        )

        return result


@pytest.fixture(scope='session')
def sort_kind():
    return 'mergesort'


default = pytest.mark.parametrize('default', [ibis.NA, ibis.literal('a')])
row_offset = pytest.mark.parametrize(
    'row_offset', list(map(ibis.literal, [-1, 1, 0]))
)
range_offset = pytest.mark.parametrize(
    'range_offset',
    [
        ibis.interval(days=1),
        2 * ibis.interval(days=1),
        -2 * ibis.interval(days=1),
    ],
)


@pytest.fixture
def row_window():
    return ibis.window(following=0, order_by='plain_int64')


@pytest.fixture
def range_window():
    return ibis.window(following=0, order_by='plain_datetimes_naive')


@pytest.fixture
def custom_window():
    return CustomWindow(
        preceding=CustomInterval(1),
        following=0,
        group_by='dup_ints',
        order_by='plain_int64',
    )


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
    expr = t.dup_strings.lead(range_offset, default=default).over(range_window)
    result = expr.execute()
    expected = (
        df[['plain_datetimes_naive', 'dup_strings']]
        .set_index('plain_datetimes_naive')
        .squeeze()
        .tshift(freq=execute(-range_offset))
        .reindex(df.plain_datetimes_naive)
        .reset_index(drop=True)
    )
    if default is not ibis.NA:
        expected = expected.fillna(execute(default))
    tm.assert_series_equal(result, expected)


@default
@range_offset
def test_lag_delta(t, df, range_offset, default, range_window):
    expr = t.dup_strings.lag(range_offset, default=default).over(range_window)
    result = expr.execute()
    expected = (
        df[['plain_datetimes_naive', 'dup_strings']]
        .set_index('plain_datetimes_naive')
        .squeeze()
        .tshift(freq=execute(range_offset))
        .reindex(df.plain_datetimes_naive)
        .reset_index(drop=True)
    )
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
        delta=(t.plain_int64 - t.plain_int64.lag())
        / (t.plain_float64 - t.plain_float64.lag()),
    )
    result = expr.execute()

    gb = df.groupby('dup_strings')
    expected = df.assign(
        last_value=gb.plain_strings.transform('last'),
        first_value=gb.plain_int64.transform('first'),
        avg_broadcast=df.plain_float64 - gb.plain_float64.transform('mean'),
        delta=(
            (df.plain_int64 - gb.plain_int64.shift(1))
            / (df.plain_float64 - gb.plain_float64.shift(1))
        ),
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
    result = expr.execute()[cols].sort_values(cols).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_batting_avg_change_in_games_per_year(players, players_df):
    expr = players.mutate(
        delta=lambda t: (t.G - t.G.lag()) / (t.yearID - t.yearID.lag())
    )

    gb = players_df.groupby('playerID')
    expected = players_df.assign(
        delta=(players_df.G - gb.G.shift(1))
        / (players_df.yearID - gb.yearID.shift(1))
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
        batting_df[['G', 'yearID']]
        .sort_values('yearID', kind=sort_kind)
        .G.expanding()
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
    more_values = (
        batting_df[columns]
        .sort_values('yearID', kind=sort_kind)
        .G.expanding()
        .sum()
        .astype('int64')
    )
    expected = batting_df.assign(more_values=more_values)
    tm.assert_frame_equal(result[expected.columns], expected)


def test_batting_cumulative_partitioned(batting, batting_df, sort_kind):
    group_by = 'playerID'
    order_by = 'yearID'

    t = batting
    expr = t.G.sum().over(
        ibis.cumulative_window(order_by=order_by, group_by=group_by)
    )
    expr = t.mutate(cumulative=expr)
    result = expr.execute()

    columns = [group_by, order_by, 'G']
    expected = (
        batting_df[columns]
        .set_index(order_by)
        .groupby(group_by)
        .G.expanding()
        .sum()
        .rename('cumulative')
    )

    tm.assert_series_equal(
        result.set_index([group_by, order_by]).sort_index().cumulative,
        expected.sort_index().astype("int64"),
    )


def test_batting_rolling(batting, batting_df, sort_kind):
    expr = batting.mutate(
        more_values=lambda t: t.G.sum().over(
            ibis.trailing_window(5, order_by=t.yearID)
        )
    )
    result = expr.execute()

    columns = ['G', 'yearID']
    more_values = (
        batting_df[columns]
        .sort_values('yearID', kind=sort_kind)
        .G.rolling(6, min_periods=1)
        .sum()
        .astype('int64')
    )
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
    expected = (
        batting_df[columns]
        .set_index(order_by)
        .groupby(group_by)
        .G.rolling(4, min_periods=1)
        .sum()
        .rename('rolled')
    )

    tm.assert_series_equal(
        result.set_index([group_by, order_by]).sort_index().rolled,
        expected.sort_index().astype("int64"),
    )


@pytest.mark.parametrize(
    'window',
    [
        ibis.window(order_by='yearID'),
        ibis.window(order_by='yearID', group_by='playerID'),
    ],
)
def test_window_failure_mode(batting, batting_df, window):
    # can't have order by without a following value of 0
    expr = batting.mutate(more_values=batting.G.sum().over(window))
    with pytest.raises(ibis.common.exceptions.OperationNotDefinedError):
        expr.execute()


def test_scalar_broadcasting(batting, batting_df):
    expr = batting.mutate(demeaned=batting.G - batting.G.mean())
    result = expr.execute()
    expected = batting_df.assign(demeaned=batting_df.G - batting_df.G.mean())
    tm.assert_frame_equal(result, expected)


def test_mutate_with_window_after_join(sort_kind):
    left_df = pd.DataFrame(
        {
            'ints': [0, 1, 2],
            'strings': ['a', 'b', 'c'],
            'dates': pd.date_range('20170101', periods=3),
        }
    )
    right_df = pd.DataFrame(
        {
            'group': [0, 1, 2] * 3,
            'value': [0, 1, np.nan, 3, 4, np.nan, 6, 7, 8],
        }
    )
    con = connect(dict(left=left_df, right=right_df))
    left, right = map(con.table, ('left', 'right'))

    joined = left.outer_join(right, left.ints == right.group)
    proj = joined[left, right.value]
    expr = proj.groupby('ints').mutate(sum=proj.value.sum())
    result = expr.execute()
    expected = pd.DataFrame(
        {
            'dates': pd.concat([left_df.dates] * 3)
            .sort_values(kind=sort_kind)
            .reset_index(drop=True),
            'ints': [0] * 3 + [1] * 3 + [2] * 3,
            'strings': ['a'] * 3 + ['b'] * 3 + ['c'] * 3,
            'value': [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, np.nan, np.nan, 8.0],
            'sum': [9.0] * 3 + [12.0] * 3 + [8.0] * 3,
        }
    )
    tm.assert_frame_equal(result[expected.columns], expected)


def test_mutate_scalar_with_window_after_join():
    left_df = pd.DataFrame({'ints': range(3)})
    right_df = pd.DataFrame(
        {
            'group': [0, 1, 2] * 3,
            'value': [0, 1, np.nan, 3, 4, np.nan, 6, 7, 8],
        }
    )
    con = connect(dict(left=left_df, right=right_df))
    left, right = map(con.table, ('left', 'right'))

    joined = left.outer_join(right, left.ints == right.group)
    proj = joined[left, right.value]
    expr = proj.mutate(sum=proj.value.sum(), const=1)
    result = expr.execute()
    expected = pd.DataFrame(
        {
            'ints': [0] * 3 + [1] * 3 + [2] * 3,
            'value': [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, np.nan, np.nan, 8.0],
            'sum': [29.0] * 9,
            'const': [1] * 9,
        }
    )
    tm.assert_frame_equal(result[expected.columns], expected)


def test_project_scalar_after_join():
    left_df = pd.DataFrame({'ints': range(3)})
    right_df = pd.DataFrame(
        {
            'group': [0, 1, 2] * 3,
            'value': [0, 1, np.nan, 3, 4, np.nan, 6, 7, 8],
        }
    )
    con = connect(dict(left=left_df, right=right_df))
    left, right = map(con.table, ('left', 'right'))

    joined = left.outer_join(right, left.ints == right.group)
    proj = joined[left, right.value]
    expr = proj[proj.value.sum().name('sum'), ibis.literal(1).name('const')]
    result = expr.execute()
    expected = pd.DataFrame({'sum': [29.0] * 9, 'const': [1] * 9})
    tm.assert_frame_equal(result[expected.columns], expected)


def test_project_list_scalar():
    df = pd.DataFrame({'ints': range(3)})
    con = connect(dict(df=df))
    expr = con.table('df')
    result = expr.mutate(res=expr.ints.quantile([0.5, 0.95])).execute()
    tm.assert_series_equal(
        result.res, pd.Series([[1.0, 1.9] for _ in range(0, 3)], name='res')
    )


@pytest.mark.parametrize(
    'index',
    [
        pytest.param(lambda time: None, id='no_index'),
        pytest.param(lambda time: time, id='index'),
    ],
)
def test_window_with_preceding_expr(index):
    time = pd.date_range('20180101', '20180110')
    start = 2
    data = np.arange(start, start + len(time))
    df = pd.DataFrame({'value': data, 'time': time}, index=index(time))
    client = connect({'df': df})
    t = client.table('df')
    expected = (
        df.set_index('time')
        .value.rolling('3d', closed='both')
        .mean()
        .reset_index(drop=True)
    )
    expected.index.name = None
    day = ibis.interval(days=1)
    window = ibis.trailing_window(3 * day, order_by=t.time)
    expr = t.value.mean().over(window)
    result = expr.execute()
    tm.assert_series_equal(result, expected)


def test_window_with_mlb():
    index = pd.date_range('20170501', '20170507')
    data = np.random.randn(len(index), 3)
    df = (
        pd.DataFrame(data, columns=list('abc'), index=index)
        .rename_axis('time')
        .reset_index(drop=False)
    )
    client = connect({'df': df})
    t = client.table('df')
    rows_with_mlb = rows_with_max_lookback(5, ibis.interval(days=10))
    expr = t.mutate(
        sum=lambda df: df.a.sum().over(
            ibis.trailing_window(rows_with_mlb, order_by='time', group_by='b')
        )
    )
    result = expr.execute()
    expected = df.set_index('time')
    gb_df = (
        expected.groupby(['b'])['a']
        .rolling('10d', closed='both')
        .apply(lambda s: s.iloc[-5:].sum(), raw=False)
        .sort_index(level=['time'])
        .reset_index(drop=True)
    )
    expected = expected.reset_index(drop=False).assign(sum=gb_df)
    tm.assert_frame_equal(result, expected)

    rows_with_mlb = rows_with_max_lookback(5, 10)
    with pytest.raises(com.IbisInputError):
        t.mutate(
            sum=lambda df: df.a.sum().over(
                ibis.trailing_window(rows_with_mlb, order_by='time')
            )
        )


def test_window_has_pre_execute_scope():
    signature = ops.Lag, PandasClient
    called = [0]

    @pre_execute.register(*signature)
    def test_pre_execute(op, client, **kwargs):
        called[0] += 1
        return Scope()

    data = {'key': list('abc'), 'value': [1, 2, 3], 'dup': list('ggh')}
    df = pd.DataFrame(data, columns=['key', 'value', 'dup'])
    client = connect(dict(df=df))
    t = client.table('df')
    window = ibis.window(order_by='value')
    expr = t.key.lag(1).over(window).name('foo')
    result = expr.execute()
    assert result is not None

    # once in window op at the top to pickup any scope changes before computing
    # twice in window op when calling execute on the ops.Lag node at the
    # beginning of execute and once before the actual computation
    assert called[0] == 3


def test_window_grouping_key_has_scope(t, df):
    param = ibis.param(dt.string)
    window = ibis.window(group_by=t.dup_strings + param)
    expr = t.plain_int64.mean().over(window)
    result = expr.execute(params={param: "a"})
    expected = df.groupby(df.dup_strings + "a").plain_int64.transform("mean")
    tm.assert_series_equal(result, expected)


def test_window_on_and_by_key_as_window_input(t, df):
    order_by = 'plain_int64'
    group_by = 'dup_ints'
    control = 'plain_float64'

    row_window = ibis.trailing_window(
        order_by=order_by, group_by=group_by, preceding=1
    )

    # Test built-in function

    tm.assert_series_equal(
        t[order_by].count().over(row_window).execute(),
        t[control].count().over(row_window).execute(),
        check_names=False,
    )

    tm.assert_series_equal(
        t[group_by].count().over(row_window).execute(),
        t[control].count().over(row_window).execute(),
        check_names=False,
    )

    # Test UDF

    @reduction(input_type=[dt.int64], output_type=dt.int64)
    def count(v):
        return len(v)

    @reduction(input_type=[dt.int64, dt.int64], output_type=dt.int64)
    def count_both(v1, v2):
        return len(v1)

    tm.assert_series_equal(
        count(t[order_by]).over(row_window).execute(),
        t[control].count().over(row_window).execute(),
        check_names=False,
    )

    tm.assert_series_equal(
        count(t[group_by]).over(row_window).execute(),
        t[control].count().over(row_window).execute(),
        check_names=False,
    )

    tm.assert_series_equal(
        count_both(t[group_by], t[order_by]).over(row_window).execute(),
        t[control].count().over(row_window).execute(),
        check_names=False,
    )


def test_custom_window_udf(t, custom_window):
    """ Test implementing  a (dummy) custom window.

    This test covers the advance use case to support custom window with udfs.

    Note that method used in this example (e.g, get_preceding, get_aggcontext)
    are unstable developer API, not stable public API.
    """

    @reduction(input_type=[dt.float64], output_type=dt.float64)
    def my_sum(v):
        return v.sum()

    # Unfortunately we cannot unregister these because singledispatch
    # doesn't support it, but this won't cause any issues either.
    @get_preceding_value.register(CustomInterval)
    def get_preceding_value_custom(preceding):
        return preceding

    @get_aggcontext.register(CustomWindow)
    def get_aggcontext_custom(
        window,
        *,
        scope,
        operand,
        parent,
        group_by,
        order_by,
        dummy_custom_window_data,
    ):
        assert dummy_custom_window_data == 'dummy_data'
        # scope and operand are not used here
        return CustomAggContext(
            parent=parent,
            group_by=group_by,
            order_by=order_by,
            output_type=operand.type(),
            max_lookback=window.max_lookback,
            preceding=window.preceding,
        )

    result = (
        my_sum(t['plain_float64'])
        .over(custom_window)
        .execute(dummy_custom_window_data='dummy_data')
    )
    expected = pd.Series([4.0, 10.0, 5.0])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'group_by,order_by',
    [
        (None, None),
        # Enable this after #2395 is merged
        # (None, 'plain_datetimes_utc'),
        ('dup_ints', None),
        ('dup_ints', 'plain_datetimes_utc'),
    ],
)
def test_rolling_window_udf_nan_and_non_numeric(t, group_by, order_by):
    # Test that rolling window can be performed on
    # (1) A column that contains NaN values
    # (2) A non-numeric column
    # (3) A non-numeric column that contains NaN value

    t = t.mutate(nan_int64=t['plain_int64'])
    t = t.mutate(nan_int64=None)

    @reduction(input_type=[dt.int64], output_type=dt.int64)
    def count_int64(v):
        return len(v)

    @reduction(input_type=[dt.timestamp], output_type=dt.int64)
    def count_timestamp(v):
        return len(v)

    @reduction(
        input_type=[t['map_of_strings_integers'].type()], output_type=dt.int64
    )
    def count_complex(v):
        return len(v)

    window = ibis.trailing_window(
        preceding=1, order_by=order_by, group_by=group_by
    )

    result_nan = count_int64(t['nan_int64']).over(window).execute()
    result_non_numeric = (
        count_timestamp(t['plain_datetimes_utc']).over(window).execute()
    )
    result_nan_non_numeric = (
        count_timestamp(t['map_of_strings_integers']).over(window).execute()
    )
    expected = t['plain_int64'].count().over(window).execute()

    tm.assert_series_equal(result_nan, expected, check_names=False)
    tm.assert_series_equal(result_non_numeric, expected, check_names=False)
    tm.assert_series_equal(result_nan_non_numeric, expected, check_names=False)
