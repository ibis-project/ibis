import dask.dataframe as dd
import pandas as pd
import pytest
from dask.dataframe.utils import tm
from pandas import Timedelta, date_range
from pytest import param

import ibis
import ibis.common.exceptions as com

# Note - computations in this file use the single threadsed scheduler (instead
# of the default multithreaded scheduler) in order to avoid a flaky interaction
# between dask and pandas in merges. There is evidence this has been fixed in
# pandas>=1.1.2 (or in other schedulers). For more background see:
# - https://github.com/dask/dask/issues/6454
# - https://github.com/dask/dask/issues/5060


join_type = pytest.mark.parametrize(
    'how',
    [
        'inner',
        'left',
        'right',
        'outer',
        param(
            'semi',
            marks=pytest.mark.xfail(
                raises=NotImplementedError, reason='Semi join not implemented'
            ),
        ),
        param(
            'anti',
            marks=pytest.mark.xfail(
                raises=NotImplementedError, reason='Anti join not implemented'
            ),
        ),
    ],
)


@join_type
def test_join(how, left, right, df1, df2):
    expr = left.join(right, left.key == right.key, how=how)[
        left, right.other_value, right.key3
    ]
    result = expr.compile()
    expected = dd.merge(df1, df2, how=how, on='key')
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


def test_cross_join(left, right, df1, df2):
    expr = left.cross_join(right)[left, right.other_value, right.key3]
    result = expr.compile()
    expected = dd.merge(
        df1.assign(dummy=1), df2.assign(dummy=1), how='inner', on='dummy'
    ).rename(columns={'key_x': 'key'})
    del expected['dummy'], expected['key_y']
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_join_project_left_table(how, left, right, df1, df2):
    expr = left.join(right, left.key == right.key, how=how)[left, right.key3]
    result = expr.compile()
    expected = dd.merge(df1, df2, how=how, on='key')[
        list(left.columns) + ['key3']
    ]
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


def test_cross_join_project_left_table(left, right, df1, df2):
    expr = left.cross_join(right)[left, right.key3]
    result = expr.compile()
    expected = dd.merge(
        df1.assign(dummy=1), df2.assign(dummy=1), how='inner', on='dummy'
    ).rename(columns={'key_x': 'key'})[list(left.columns) + ['key3']]
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_join_with_multiple_predicates(how, left, right, df1, df2):
    expr = left.join(
        right, [left.key == right.key, left.key2 == right.key3], how=how
    )[left, right.key3, right.other_value]
    result = expr.compile()
    expected = dd.merge(
        df1, df2, how=how, left_on=['key', 'key2'], right_on=['key', 'key3']
    ).reset_index(drop=True)
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_join_with_multiple_predicates_written_as_one(
    how, left, right, df1, df2
):
    predicate = (left.key == right.key) & (left.key2 == right.key3)
    expr = left.join(right, predicate, how=how)[
        left, right.key3, right.other_value
    ]
    result = expr.compile()
    expected = dd.merge(
        df1, df2, how=how, left_on=['key', 'key2'], right_on=['key', 'key3']
    ).reset_index(drop=True)
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_join_with_invalid_predicates(how, left, right):
    predicate = (left.key == right.key) & (left.key2 <= right.key3)
    expr = left.join(right, predicate, how=how)
    with pytest.raises(TypeError):
        expr.compile()

    predicate = left.key >= right.key
    expr = left.join(right, predicate, how=how)
    with pytest.raises(TypeError):
        expr.compile()


@join_type
@pytest.mark.xfail(reason='Hard to detect this case')
def test_join_with_duplicate_non_key_columns(how, left, right, df1, df2):
    left = left.mutate(x=left.value * 2)
    right = right.mutate(x=right.other_value * 3)
    expr = left.join(right, left.key == right.key, how=how)

    # This is undefined behavior because `x` is duplicated. This is difficult
    # to detect
    with pytest.raises(ValueError):
        expr.compile()


@join_type
def test_join_with_duplicate_non_key_columns_not_selected(
    how, left, right, df1, df2
):
    left = left.mutate(x=left.value * 2)
    right = right.mutate(x=right.other_value * 3)
    right = right[['key', 'other_value']]
    expr = left.join(right, left.key == right.key, how=how)[
        left, right.other_value
    ]
    result = expr.compile()
    expected = dd.merge(
        df1.assign(x=df1.value * 2),
        df2[['key', 'other_value']],
        how=how,
        on='key',
    )
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_join_with_post_expression_selection(how, left, right, df1, df2):
    join = left.join(right, left.key == right.key, how=how)
    expr = join[left.key, left.value, right.other_value]
    result = expr.compile()
    expected = dd.merge(df1, df2, on='key', how=how)[
        ['key', 'value', 'other_value']
    ]
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_join_with_post_expression_filter(how, left):
    lhs = left[['key', 'key2']]
    rhs = left[['key2', 'value']]

    joined = lhs.join(rhs, 'key2', how=how)
    projected = joined[lhs, rhs.value]
    expr = projected[projected.value == 4]
    result = expr.compile()

    df1 = lhs.compile()
    df2 = rhs.compile()
    expected = dd.merge(df1, df2, on='key2', how=how)
    expected = expected.loc[expected.value == 4].reset_index(drop=True)

    tm.assert_frame_equal(
        result.compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_multi_join_with_post_expression_filter(how, left, df1):
    lhs = left[['key', 'key2']]
    rhs = left[['key2', 'value']]
    rhs2 = left[['key2', 'value']].relabel({'value': 'value2'})

    joined = lhs.join(rhs, 'key2', how=how)
    projected = joined[lhs, rhs.value]
    filtered = projected[projected.value == 4]

    joined2 = filtered.join(rhs2, 'key2')
    projected2 = joined2[filtered.key, rhs2.value2]
    expr = projected2[projected2.value2 == 3]

    result = expr.compile()

    df1 = lhs.compile()
    df2 = rhs.compile()
    df3 = rhs2.compile()
    expected = dd.merge(df1, df2, on='key2', how=how)
    expected = expected.loc[expected.value == 4].reset_index(drop=True)
    expected = dd.merge(expected, df3, on='key2')[['key', 'value2']]
    expected = expected.loc[expected.value2 == 3].reset_index(drop=True)

    tm.assert_frame_equal(
        result.compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@pytest.mark.xfail(reason="TODO - execute_join - #2553")
@join_type
def test_join_with_non_trivial_key(how, left, right, df1, df2):
    # also test that the order of operands in the predicate doesn't matter
    join = left.join(right, right.key.length() == left.key.length(), how=how)
    expr = join[left.key, left.value, right.other_value]
    result = expr.compile()

    expected = (
        dd.merge(
            df1.assign(key_len=df1.key.str.len()),
            df2.assign(key_len=df2.key.str.len()),
            on='key_len',
            how=how,
        )
        .drop(['key_len', 'key_y', 'key2', 'key3'], axis=1)
        .rename(columns={'key_x': 'key'})
    )
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@pytest.mark.xfail(reason="TODO - execute_join - #2553")
@join_type
def test_join_with_non_trivial_key_project_table(how, left, right, df1, df2):
    # also test that the order of operands in the predicate doesn't matter
    join = left.join(right, right.key.length() == left.key.length(), how=how)
    expr = join[left, right.other_value]
    expr = expr[expr.key.length() == 1]
    result = expr.compile()

    expected = (
        dd.merge(
            df1.assign(key_len=df1.key.str.len()),
            df2.assign(key_len=df2.key.str.len()),
            on='key_len',
            how=how,
        )
        .drop(['key_len', 'key_y', 'key2', 'key3'], axis=1)
        .rename(columns={'key_x': 'key'})
    )
    expected = expected.loc[expected.key.str.len() == 1]
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@join_type
def test_join_with_project_right_duplicate_column(client, how, left, df1, df3):
    # also test that the order of operands in the predicate doesn't matter
    right = client.table('df3')
    join = left.join(right, ['key'], how=how)
    expr = join[left.key, right.key2, right.other_value]
    result = expr.compile()

    expected = (
        dd.merge(df1, df3, on='key', how=how)
        .drop(['key2_x', 'key3', 'value'], axis=1)
        .rename(columns={'key2_y': 'key2'})
    )
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


def test_join_with_window_function(
    players_base, players_df, batting, batting_df
):
    players = players_base

    # this should be semi_join
    tbl = batting.left_join(players, ['playerID'])
    t = tbl[batting.G, batting.playerID, batting.teamID]
    expr = t.groupby(t.teamID).mutate(
        team_avg=lambda d: d.G.mean(),
        demeaned_by_player=lambda d: d.G - d.G.mean(),
    )
    result = expr.compile()

    expected = dd.merge(
        batting_df, players_df[['playerID']], on='playerID', how='left'
    )[['G', 'playerID', 'teamID']]
    team_avg = expected.groupby('teamID').G.transform('mean')
    expected = expected.assign(
        team_avg=team_avg, demeaned_by_player=lambda df: df.G - team_avg
    )

    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


merge_asof_minversion = pytest.mark.skipif(
    pd.__version__ < '0.19.2',
    reason="at least pandas-0.19.2 required for merge_asof",
)


@merge_asof_minversion
def test_asof_join(time_left, time_right, time_df1, time_df2):
    expr = time_left.asof_join(time_right, 'time')[
        time_left, time_right.other_value
    ]
    result = expr.compile()
    expected = dd.merge_asof(time_df1, time_df2, on='time')
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@merge_asof_minversion
def test_asof_join_predicate(time_left, time_right, time_df1, time_df2):
    expr = time_left.asof_join(time_right, time_left.time == time_right.time)[
        time_left, time_right.other_value
    ]
    result = expr.compile()
    expected = dd.merge_asof(time_df1, time_df2, on='time')
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@merge_asof_minversion
def test_keyed_asof_join(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(time_keyed_right, 'time', by='key')[
        time_keyed_left, time_keyed_right.other_value
    ]
    result = expr.compile()
    expected = dd.merge_asof(
        time_keyed_df1, time_keyed_df2, on='time', by='key'
    )
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@merge_asof_minversion
def test_keyed_asof_join_with_tolerance(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(
        time_keyed_right, 'time', by='key', tolerance=2 * ibis.interval(days=1)
    )[time_keyed_left, time_keyed_right.other_value]
    result = expr.compile()
    expected = dd.merge_asof(
        time_keyed_df1,
        time_keyed_df2,
        on='time',
        by='key',
        tolerance=Timedelta('2D'),
    )
    tm.assert_frame_equal(
        result[expected.columns].compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@pytest.mark.parametrize(
    "how",
    [
        "left",
        pytest.param(
            "right",
            marks=pytest.mark.xfail(
                raises=AttributeError, reason="right_join is not an ibis API"
            ),
        ),
        "inner",
        "outer",
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda join: join["a0", "a1"], id="tuple"),
        pytest.param(lambda join: join[["a0", "a1"]], id="list"),
        pytest.param(lambda join: join.select(["a0", "a1"]), id="select"),
    ],
)
@pytest.mark.xfail(
    raises=(com.IbisError, AttributeError),
    reason="Select from unambiguous joins not implemented",
)
def test_select_on_unambiguous_join(how, func, npartitions):
    df_t = dd.from_pandas(
        pd.DataFrame({'a0': [1, 2, 3], 'b1': list("aab")}),
        npartitions=npartitions,
    )
    df_s = dd.from_pandas(
        pd.DataFrame({'a1': [2, 3, 4], 'b2': list("abc")}),
        npartitions=npartitions,
    )
    con = ibis.dask.connect({"t": df_t, "s": df_s})
    t = con.table("t")
    s = con.table("s")
    method = getattr(t, f"{how}_join")
    join = method(s, t.b1 == s.b2)
    expected = dd.merge(df_t, df_s, left_on=["b1"], right_on=["b2"], how=how)[
        ["a0", "a1"]
    ]
    assert not expected.compute(scheduler='single-threaded').empty
    expr = func(join)
    result = expr.compile()
    tm.assert_frame_equal(
        result.compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda join: join["a0", "a1"], id="tuple"),
        pytest.param(lambda join: join[["a0", "a1"]], id="list"),
        pytest.param(lambda join: join.select(["a0", "a1"]), id="select"),
    ],
)
@pytest.mark.xfail(
    raises=(com.IbisError, AttributeError),
    reason="Select from unambiguous joins not implemented",
)
@merge_asof_minversion
def test_select_on_unambiguous_asof_join(func, npartitions):
    df_t = dd.from_pandas(
        pd.DataFrame(
            {'a0': [1, 2, 3], 'b1': date_range("20180101", periods=3)}
        ),
        npartitions=npartitions,
    )
    df_s = dd.from_pandas(
        pd.DataFrame(
            {'a1': [2, 3, 4], 'b2': date_range("20171230", periods=3)}
        ),
        npartitions=npartitions,
    )
    con = ibis.dask.connect({"t": df_t, "s": df_s})
    t = con.table("t")
    s = con.table("s")
    join = t.asof_join(s, t.b1 == s.b2)
    expected = dd.merge_asof(df_t, df_s, left_on=["b1"], right_on=["b2"])[
        ["a0", "a1"]
    ]
    assert not expected.compute(scheduler='single-threaded').empty
    expr = func(join)
    result = expr.compile()
    tm.assert_frame_equal(
        result.compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )


def test_materialized_join(npartitions):
    df = dd.from_pandas(
        pd.DataFrame({"test": [1, 2, 3], "name": ["a", "b", "c"]}),
        npartitions=npartitions,
    )
    df_2 = dd.from_pandas(
        pd.DataFrame({"test_2": [1, 5, 6], "name_2": ["d", "e", "f"]}),
        npartitions=npartitions,
    )

    conn = ibis.dask.connect({"df": df, "df_2": df_2})

    ibis_table_1 = conn.table("df")
    ibis_table_2 = conn.table("df_2")

    joined = ibis_table_1.outer_join(
        ibis_table_2,
        predicates=ibis_table_1["test"] == ibis_table_2["test_2"],
    )
    joined = joined.materialize()
    result = joined.compile()
    expected = dd.merge(
        df,
        df_2,
        left_on="test",
        right_on="test_2",
        how="outer",
    )
    tm.assert_frame_equal(
        result.compute(scheduler='single-threaded'),
        expected.compute(scheduler='single-threaded'),
    )
