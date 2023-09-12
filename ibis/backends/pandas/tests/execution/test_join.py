from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis

# SEMI and ANTI are checked in backend tests
mutating_join_type = pytest.mark.parametrize(
    "how",
    ["inner", "left", "right", "outer"],
)


@mutating_join_type
def test_join(how, left, right, df1, df2):
    expr = left.join(right, left.key == right.key, how=how)[
        left, right.other_value, right.key3
    ]
    result = expr.execute()
    expected = pd.merge(df1, df2, how=how, on="key")
    tm.assert_frame_equal(result[expected.columns], expected)


def test_cross_join(left, right, df1, df2):
    expr = left.cross_join(right)[left, right.other_value, right.key3]
    result = expr.execute()
    expected = pd.merge(
        df1.assign(dummy=1), df2.assign(dummy=1), how="inner", on="dummy"
    ).rename(columns={"key_x": "key"})
    del expected["dummy"], expected["key_y"]
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_project_left_table(how, left, right, df1, df2):
    expr = left.join(right, left.key == right.key, how=how)[left, right.key3]
    result = expr.execute()
    expected = pd.merge(df1, df2, how=how, on="key")[list(left.columns) + ["key3"]]
    tm.assert_frame_equal(result[expected.columns], expected)


def test_cross_join_project_left_table(left, right, df1, df2):
    expr = left.cross_join(right)[left, right.key3]
    result = expr.execute()
    expected = pd.merge(
        df1.assign(dummy=1), df2.assign(dummy=1), how="inner", on="dummy"
    ).rename(columns={"key_x": "key"})[list(left.columns) + ["key3"]]
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_with_multiple_predicates(how, left, right, df1, df2):
    expr = left.join(right, [left.key == right.key, left.key2 == right.key3], how=how)[
        left, right.key3, right.other_value
    ]
    result = expr.execute()
    expected = pd.merge(
        df1, df2, how=how, left_on=["key", "key2"], right_on=["key", "key3"]
    ).reset_index(drop=True)
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_with_multiple_predicates_written_as_one(how, left, right, df1, df2):
    predicate = (left.key == right.key) & (left.key2 == right.key3)
    expr = left.join(right, predicate, how=how)[left, right.key3, right.other_value]
    result = expr.execute()
    expected = pd.merge(
        df1, df2, how=how, left_on=["key", "key2"], right_on=["key", "key3"]
    ).reset_index(drop=True)
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_with_invalid_predicates(how, left, right):
    predicate = (left.key == right.key) & (left.key2 <= right.key3)
    expr = left.join(right, predicate, how=how)
    with pytest.raises(TypeError):
        expr.execute()

    predicate = left.key >= right.key
    expr = left.join(right, predicate, how=how)
    with pytest.raises(TypeError):
        expr.execute()


@mutating_join_type
@pytest.mark.xfail(reason="Hard to detect this case")
def test_join_with_duplicate_non_key_columns(how, left, right):
    left = left.mutate(x=left.value * 2)
    right = right.mutate(x=right.other_value * 3)
    expr = left.join(right, left.key == right.key, how=how)

    # This is undefined behavior because `x` is duplicated. This is difficult
    # to detect
    with pytest.raises(ValueError):
        expr.execute()


@mutating_join_type
def test_join_with_duplicate_non_key_columns_not_selected(how, left, right, df1, df2):
    left = left.mutate(x=left.value * 2)
    right = right.mutate(x=right.other_value * 3)
    right = right[["key", "other_value"]]
    expr = left.join(right, left.key == right.key, how=how)[left, right.other_value]
    result = expr.execute()
    expected = pd.merge(
        df1.assign(x=df1.value * 2),
        df2[["key", "other_value"]],
        how=how,
        on="key",
    )
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_with_post_expression_selection(how, left, right, df1, df2):
    join = left.join(right, left.key == right.key, how=how)
    expr = join[left.key, left.value, right.other_value]
    result = expr.execute()
    expected = pd.merge(df1, df2, on="key", how=how)[["key", "value", "other_value"]]
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_with_post_expression_filter(how, left):
    lhs = left[["key", "key2"]]
    rhs = left[["key2", "value"]]

    joined = lhs.join(rhs, "key2", how=how)
    projected = joined[lhs, rhs.value]
    expr = projected[projected.value == 4]
    result = expr.execute()

    df1 = lhs.execute()
    df2 = rhs.execute()
    expected = pd.merge(df1, df2, on="key2", how=how)
    expected = expected.loc[expected.value == 4].reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@mutating_join_type
def test_multi_join_with_post_expression_filter(how, left, df1):
    lhs = left[["key", "key2"]]
    rhs = left[["key2", "value"]]
    rhs2 = left[["key2", "value"]].rename(value2="value")

    joined = lhs.join(rhs, "key2", how=how)
    projected = joined[lhs, rhs.value]
    filtered = projected[projected.value == 4]

    joined2 = filtered.join(rhs2, "key2")
    projected2 = joined2[filtered.key, rhs2.value2]
    expr = projected2[projected2.value2 == 3]

    result = expr.execute()

    df1 = lhs.execute()
    df2 = rhs.execute()
    df3 = rhs2.execute()
    expected = pd.merge(df1, df2, on="key2", how=how)
    expected = expected.loc[expected.value == 4].reset_index(drop=True)
    expected = pd.merge(expected, df3, on="key2")[["key", "value2"]]
    expected = expected.loc[expected.value2 == 3].reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@mutating_join_type
def test_join_with_non_trivial_key(how, left, right, df1, df2):
    # also test that the order of operands in the predicate doesn't matter
    join = left.join(right, right.key.length() == left.key.length(), how=how)
    expr = join[left.key, left.value, right.other_value]
    result = expr.execute()

    expected = (
        pd.merge(
            df1.assign(key_len=df1.key.str.len()),
            df2.assign(key_len=df2.key.str.len()),
            on="key_len",
            how=how,
        )
        .drop(["key_len", "key_y", "key2", "key3"], axis=1)
        .rename(columns={"key_x": "key"})
    )
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_with_non_trivial_key_project_table(how, left, right, df1, df2):
    # also test that the order of operands in the predicate doesn't matter
    join = left.join(right, right.key.length() == left.key.length(), how=how)
    expr = join[left, right.other_value]
    expr = expr[expr.key.length() == 1]
    result = expr.execute()

    expected = (
        pd.merge(
            df1.assign(key_len=df1.key.str.len()),
            df2.assign(key_len=df2.key.str.len()),
            on="key_len",
            how=how,
        )
        .drop(["key_len", "key_y", "key2", "key3"], axis=1)
        .rename(columns={"key_x": "key"})
    )
    expected = expected.loc[expected.key.str.len() == 1]
    tm.assert_frame_equal(result[expected.columns], expected)


@mutating_join_type
def test_join_with_project_right_duplicate_column(client, how, left, df1, df3):
    # also test that the order of operands in the predicate doesn't matter
    right = client.table("df3")
    join = left.join(right, ["key"], how=how)
    expr = join[left.key, right.key2, right.other_value]
    result = expr.execute()

    expected = (
        pd.merge(df1, df3, on="key", how=how)
        .drop(["key2_x", "key3", "value"], axis=1)
        .rename(columns={"key2_y": "key2"})
    )
    tm.assert_frame_equal(result[expected.columns], expected)


def test_join_with_window_function(players_base, players_df, batting, batting_df):
    players = players_base

    # this should be semi_join
    tbl = batting.left_join(players, ["playerID"])
    t = tbl[batting.G, batting.playerID, batting.teamID]
    expr = t.group_by(t.teamID).mutate(
        team_avg=lambda d: d.G.mean(),
        demeaned_by_player=lambda d: d.G - d.G.mean(),
    )
    result = expr.execute()

    expected = pd.merge(
        batting_df, players_df[["playerID"]], on="playerID", how="left"
    )[["G", "playerID", "teamID"]]
    team_avg = expected.groupby("teamID").G.transform("mean")
    expected = expected.assign(
        team_avg=team_avg, demeaned_by_player=lambda df: df.G - team_avg
    )

    tm.assert_frame_equal(result[expected.columns], expected)


merge_asof_minversion = pytest.mark.skipif(
    pd.__version__ < "0.19.2",
    reason="at least pandas-0.19.2 required for merge_asof",
)


@merge_asof_minversion
def test_asof_join(time_left, time_right, time_df1, time_df2):
    expr = time_left.asof_join(time_right, "time")
    result = expr.execute()
    expected = pd.merge_asof(time_df1, time_df2, on="time")
    tm.assert_frame_equal(result[expected.columns], expected)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])


@merge_asof_minversion
def test_asof_join_predicate(time_left, time_right, time_df1, time_df2):
    expr = time_left.asof_join(time_right, time_left.time == time_right.time)
    result = expr.execute()
    expected = pd.merge_asof(time_df1, time_df2, on="time")
    tm.assert_frame_equal(result[expected.columns], expected)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])


@merge_asof_minversion
def test_keyed_asof_join(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(time_keyed_right, "time", by="key")
    result = expr.execute()
    expected = pd.merge_asof(time_keyed_df1, time_keyed_df2, on="time", by="key")
    tm.assert_frame_equal(result[expected.columns], expected)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["key"], result["key_right"])


@merge_asof_minversion
def test_keyed_asof_join_with_tolerance(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    expr = time_keyed_left.asof_join(
        time_keyed_right, "time", by="key", tolerance=2 * ibis.interval(days=1)
    )
    result = expr.execute()
    expected = pd.merge_asof(
        time_keyed_df1,
        time_keyed_df2,
        on="time",
        by="key",
        tolerance=pd.Timedelta("2D"),
    )
    tm.assert_frame_equal(result[expected.columns], expected)
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["time"], result["time_right"])
    with pytest.raises(AssertionError):
        tm.assert_series_equal(result["key"], result["key_right"])


@merge_asof_minversion
def test_asof_join_overlapping_non_predicate(
    time_keyed_left, time_keyed_right, time_keyed_df1, time_keyed_df2
):
    # Add a junk column with a colliding name
    time_keyed_left = time_keyed_left.mutate(
        collide=time_keyed_left.key + time_keyed_left.value
    )
    time_keyed_right = time_keyed_right.mutate(
        collide=time_keyed_right.key + time_keyed_right.other_value
    )
    time_keyed_df1.assign(collide=time_keyed_df1["key"] + time_keyed_df1["value"])
    time_keyed_df2.assign(collide=time_keyed_df2["key"] + time_keyed_df2["other_value"])

    expr = time_keyed_left.asof_join(
        time_keyed_right, predicates=[("time", "time")], by=[("key", "key")]
    )
    result = expr.execute()
    expected = pd.merge_asof(
        time_keyed_df1, time_keyed_df2, on="time", by="key", suffixes=("", "_right")
    )
    tm.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize(
    "how",
    [
        "left",
        "right",
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
def test_select_on_unambiguous_join(how, func):
    df_t = pd.DataFrame({"a0": [1, 2, 3], "b1": list("aab")})
    df_s = pd.DataFrame({"a1": [2, 3, 4], "b2": list("abc")})
    con = ibis.pandas.connect({"t": df_t, "s": df_s})
    t = con.table("t")
    s = con.table("s")
    method = getattr(t, f"{how}_join")
    join = method(s, t.b1 == s.b2)
    expected = pd.merge(df_t, df_s, left_on=["b1"], right_on=["b2"], how=how)[
        ["a0", "a1"]
    ]
    assert not expected.empty
    expr = func(join)
    result = expr.execute()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda join: join["a0", "a1"], id="tuple"),
        pytest.param(lambda join: join[["a0", "a1"]], id="list"),
        pytest.param(lambda join: join.select(["a0", "a1"]), id="select"),
    ],
)
@merge_asof_minversion
def test_select_on_unambiguous_asof_join(func):
    df_t = pd.DataFrame({"a0": [1, 2, 3], "b1": pd.date_range("20180101", periods=3)})
    df_s = pd.DataFrame({"a1": [2, 3, 4], "b2": pd.date_range("20171230", periods=3)})
    con = ibis.pandas.connect({"t": df_t, "s": df_s})
    t = con.table("t")
    s = con.table("s")
    join = t.asof_join(s, t.b1 == s.b2)
    expected = pd.merge_asof(df_t, df_s, left_on=["b1"], right_on=["b2"])[["a0", "a1"]]
    assert not expected.empty
    expr = func(join)
    result = expr.execute()
    tm.assert_frame_equal(result, expected)


def test_outer_join():
    df = pd.DataFrame({"test": [1, 2, 3], "name": ["a", "b", "c"]})
    df_2 = pd.DataFrame({"test_2": [1, 5, 6], "name_2": ["d", "e", "f"]})

    conn = ibis.pandas.connect({"df": df, "df_2": df_2})

    ibis_table_1 = conn.table("df")
    ibis_table_2 = conn.table("df_2")

    joined = ibis_table_1.outer_join(
        ibis_table_2,
        predicates=ibis_table_1["test"] == ibis_table_2["test_2"],
    )
    result = joined.execute()
    expected = pd.merge(
        df,
        df_2,
        left_on="test",
        right_on="test_2",
        how="outer",
    )
    tm.assert_frame_equal(result, expected)


def test_mutate_after_join():
    # GH3090
    df = pd.DataFrame(
        {
            "p_Order_Priority": ["C", "H", "L", "M"],
            "p_count": [9, 9, 15, 11],
            "p_density": [0.204545, 0.204545, 0.340909, 0.250000],
        }
    )
    df_2 = pd.DataFrame(
        {
            "q_Order_Priority": ["C", "H", "L", "M"],
            "q_count": [13, 21, 12, 10],
            "q_density": [0.232143, 0.375000, 0.214286, 0.178571],
        }
    )

    conn = ibis.pandas.connect({"df": df, "df_2": df_2})

    ibis_table_1 = conn.table("df")
    ibis_table_2 = conn.table("df_2")

    joined = ibis_table_1.outer_join(
        ibis_table_2,
        predicates=(
            ibis_table_1["p_Order_Priority"] == ibis_table_2["q_Order_Priority"]
        ),
    )

    joined = joined.mutate(
        bins=(
            joined["p_Order_Priority"]
            .isnull()
            .ifelse(joined["q_Order_Priority"], joined["p_Order_Priority"])
        ),
        p_count=joined["p_count"].fillna(0),
        q_count=joined["q_count"].fillna(0),
        p_density=joined.p_density.fillna(1e-10),
        q_density=joined.q_density.fillna(1e-10),
        features="Order_Priority",
    )

    expected = pd.DataFrame(
        {
            "p_Order_Priority": list("CHLM"),
            "p_count": [9, 9, 15, 11],
            "p_density": [0.204545, 0.204545, 0.340909, 0.250000],
            "q_Order_Priority": list("CHLM"),
            "q_count": [13, 21, 12, 10],
            "q_density": [0.232143, 0.375000, 0.214286, 0.178571],
            "bins": list("CHLM"),
            "features": ["Order_Priority"] * 4,
        }
    )
    result = joined.execute()
    tm.assert_frame_equal(result, expected)


@pytest.fixture
def tracts_df():
    return pd.DataFrame(
        [[1, 1], [2, 1], [3, 2], [4, 2], [5, 3], [6, 4]],
        columns=["tract_id", "tract_farm_id"],
    )


@pytest.fixture
def fields_df():
    vals = [
        [1, 1, "[(0, 2), (1, 3), (2, 0), (3, 1)]"],
        [2, 1, "[(2, 2), (3, 2), (3, 1)]"],
        [3, 2, "[(0, 1), (-1, 0), (-2, 0), (-2, 1)]"],
        [4, 3, "[(0, 1), (1, 1), (1, 2), (0, 2)]"],
        [5, 3, "[(1, 0), (2, 0), (2, 3), (1, 3)]"],
        [6, 3, "[(2, 0), (3, 0), (3, 2), (2, 2)]"],
        [7, 4, "[(-1, -1), (0, -1), (0, -2)]"],
        [8, 4, "[(1, 0), (1, -2), (0, -2), (0, -1)]"],
        [
            9,
            5,
            str(
                [
                    (1, 0),
                    (2, 0),
                    (1, -1),
                    (1, -2),
                    (-1, -2),
                    (-1, -1),
                    (-2, 0),
                    (-1, 0),
                    (0, -1),
                ]
            ),
        ],
        [10, 6, "[(-1, 2), (0, 2), (0, 0), (-1, 0)]"],
        [11, 6, "[(0, 2), (1, 2), (1, 1), (0, 1)]"],
    ]
    return pd.DataFrame(
        vals,
        columns=["field_id", "field_tract_id", "field_vertices"],
    )


@pytest.fixture
def harvest_df():
    vals = [
        [1, 1, 1, 1, 1, 65.80],
        [2, 2, 1, 2, 2, 5750.00],
        [3, 3, 1, 1, 1, 59.85],
        [4, 4, 2, 2, 2, 10100.00],
        [5, 5, 2, 1, 1, 90.30],
        [6, 6, 2, 2, 2, 21000.00],
        [7, 7, 2, 2, 2, 5150.00],
        [8, 8, 2, 1, 1, 53.55],
        [9, 9, 3, 1, 1, 147.00],
        [10, 10, 4, 1, 1, 70.70],
        [11, 11, 4, 2, 2, 9600.00],
        [12, 1, 1, 2, 4, 22800.00],
        [13, 2, 1, 1, 3, 19.25],
        [14, 3, 1, 2, 4, 13050.00],
        [15, 4, 2, 1, 3, 31.15],
        [16, 5, 2, 2, 4, 33000.00],
        [17, 6, 2, 1, 3, 64.40],
        [18, 7, 2, 1, 3, 16.45],
        [19, 8, 2, 2, 4, 15000.00],
        [20, 9, 3, 2, 4, 38400.00],
        [21, 10, 4, 2, 4, 19800.00],
        [22, 11, 4, 1, 3, 34.30],
    ]

    return pd.DataFrame(
        vals,
        columns=[
            "harvest_id",
            "harvest_field_id",
            "harvest_farmer_group_id",
            "harvest_crop_id",
            "harvest_date_id",
            "harvest_value",
        ],
    )


def test_multijoin(tracts_df, fields_df, harvest_df):
    conn = ibis.pandas.connect(
        dict(
            tracts=tracts_df,
            fields=fields_df,
            harvest=harvest_df,
        )
    )

    tracts, fields, harvest = map(conn.table, "tracts fields harvest".split())

    fielded = harvest.inner_join(
        fields,
        harvest.harvest_field_id == fields.field_id,
    )
    tracted = fielded.inner_join(
        tracts,
        fielded.field_tract_id == tracts.tract_id,
    )
    result = tracted.execute()

    fielded_df = pd.merge(
        harvest_df,
        fields_df,
        left_on="harvest_field_id",
        right_on="field_id",
    )
    expected = pd.merge(
        fielded_df,
        tracts_df,
        left_on="field_tract_id",
        right_on="tract_id",
    )

    tm.assert_frame_equal(result, expected)
