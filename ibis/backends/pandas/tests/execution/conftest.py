from __future__ import annotations

import decimal

import numpy as np
import pandas as pd
import pytest

import ibis.expr.datatypes as dt
from ibis.backends.pandas import Backend


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        {
            "plain_int64": list(range(1, 4)),
            "plain_strings": list("abc"),
            "plain_float64": [4.0, 5.0, 6.0],
            "plain_datetimes_naive": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=3).values
            ),
            "plain_datetimes_ny": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=3).values
            ).dt.tz_localize("America/New_York"),
            "plain_datetimes_utc": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=3).values
            ).dt.tz_localize("UTC"),
            "plain_uint64": pd.Series(range(1, 4), dtype=np.dtype("uint64")),
            "dup_strings": list("dad"),
            "dup_ints": [1, 2, 1],
            "float64_as_strings": ["100.01", "234.23", "-999.34"],
            "int64_as_strings": list(map(str, range(1, 4))),
            "strings_with_space": [" ", "abab", "ddeeffgg"],
            "translate_from_strings": ["rmz", "abc", "ghj"],
            "translate_to_strings": ["lns", "ovk", "jfr"],
            "int64_with_zeros": [0, 1, 0],
            "float64_with_zeros": [1.0, 0.0, 1.0],
            "float64_positive": [1.0, 2.0, 1.0],
            "strings_with_nulls": ["a", None, "b"],
            "datetime_strings_naive": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=3).values
            ).astype(str),
            "datetime_strings_ny": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=3).values
            )
            .dt.tz_localize("America/New_York")
            .astype(str),
            "datetime_strings_utc": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=3).values
            )
            .dt.tz_localize("UTC")
            .astype(str),
            "decimal": list(map(decimal.Decimal, ["1.0", "2", "3.234"])),
            "array_of_float64": [
                np.array([1.0, 2.0], dtype="float64"),
                np.array([3.0], dtype="float64"),
                np.array([], dtype="float64"),
            ],
            "array_of_int64": [
                np.array([1, 2], dtype="int64"),
                np.array([], dtype="int64"),
                np.array([3], dtype="int64"),
            ],
            "array_of_strings": [
                np.array(["a", "b"], dtype="object"),
                np.array([], dtype="object"),
                np.array(["c"], dtype="object"),
            ],
            "map_of_strings_integers": [{"a": 1, "b": 2}, None, {}],
            "map_of_integers_strings": [{}, None, {1: "a", 2: "b"}],
            "map_of_complex_values": [None, {"a": [1, 2, 3], "b": []}, {}],
        }
    )


@pytest.fixture(scope="module")
def batting_df(data_dir):
    num_rows = 1000
    start_index = 30
    df = pd.read_parquet(data_dir / "parquet" / "batting.parquet").iloc[
        start_index : start_index + num_rows
    ]
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def awards_players_df(data_dir):
    return pd.read_parquet(data_dir / "parquet" / "awards_players.parquet")


@pytest.fixture(scope="module")
def df1():
    return pd.DataFrame(
        {"key": list("abcd"), "value": [3, 4, 5, 6], "key2": list("eeff")}
    )


@pytest.fixture(scope="module")
def df2():
    return pd.DataFrame(
        {"key": list("ac"), "other_value": [4.0, 6.0], "key3": list("fe")}
    )


@pytest.fixture(scope="module")
def intersect_df2():
    return pd.DataFrame({"key": list("cd"), "value": [5, 6], "key2": list("ff")})


@pytest.fixture(scope="module")
def time_df1():
    return pd.DataFrame(
        {"time": pd.to_datetime([1, 2, 3, 4]), "value": [1.1, 2.2, 3.3, 4.4]}
    )


@pytest.fixture(scope="module")
def time_df2():
    return pd.DataFrame({"time": pd.to_datetime([2, 4]), "other_value": [1.2, 2.0]})


@pytest.fixture(scope="module")
def time_df3():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=8).values
            ),
            "id": list(range(1, 5)) * 2,
            "value": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
        }
    )


@pytest.fixture(scope="module")
def time_keyed_df1():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=6).values
            ),
            "key": [1, 2, 3, 1, 2, 3],
            "value": [1.2, 1.4, 2.0, 4.0, 8.0, 16.0],
        }
    )


@pytest.fixture(scope="module")
def time_keyed_df2():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(
                    start="2017-01-02 01:02:03.234", freq="3D", periods=3
                ).values
            ),
            "key": [1, 2, 3],
            "other_value": [1.1, 1.2, 2.2],
        }
    )


@pytest.fixture(scope="module")
def client(
    df,
    df1,
    df2,
    df3,
    time_df1,
    time_df2,
    time_df3,
    time_keyed_df1,
    time_keyed_df2,
    intersect_df2,
):
    return Backend().connect(
        {
            "df": df,
            "df1": df1,
            "df2": df2,
            "df3": df3,
            "left": df1,
            "right": df2,
            "time_df1": time_df1,
            "time_df2": time_df2,
            "time_df3": time_df3,
            "time_keyed_df1": time_keyed_df1,
            "time_keyed_df2": time_keyed_df2,
            "intersect_df2": intersect_df2,
        }
    )


@pytest.fixture(scope="module")
def df3():
    return pd.DataFrame(
        {
            "key": list("ac"),
            "other_value": [4.0, 6.0],
            "key2": list("ae"),
            "key3": list("fe"),
        }
    )


t_schema = {
    "decimal": dt.Decimal(4, 3),
    "array_of_float64": dt.Array(dt.double),
    "array_of_int64": dt.Array(dt.int64),
    "array_of_strings": dt.Array(dt.string),
    "map_of_strings_integers": dt.Map(dt.string, dt.int64),
    "map_of_integers_strings": dt.Map(dt.int64, dt.string),
    "map_of_complex_values": dt.Map(dt.string, dt.Array(dt.int64)),
}


@pytest.fixture(scope="module")
def t(client):
    return client.table("df", schema=t_schema)


@pytest.fixture(scope="module")
def lahman(batting_df, awards_players_df):
    return Backend().connect(
        {"batting": batting_df, "awards_players": awards_players_df}
    )


@pytest.fixture(scope="module")
def left(client):
    return client.table("left")


@pytest.fixture(scope="module")
def right(client):
    return client.table("right")


@pytest.fixture(scope="module")
def time_left(client):
    return client.table("time_df1")


@pytest.fixture(scope="module")
def time_right(client):
    return client.table("time_df2")


@pytest.fixture(scope="module")
def time_table(client):
    return client.table("time_df3")


@pytest.fixture(scope="module")
def time_keyed_left(client):
    return client.table("time_keyed_df1")


@pytest.fixture(scope="module")
def time_keyed_right(client):
    return client.table("time_keyed_df2")


@pytest.fixture(scope="module")
def batting(lahman):
    return lahman.table("batting")


@pytest.fixture(scope="module")
def sel_cols(batting):
    cols = batting.columns
    start, end = cols.index("AB"), cols.index("H") + 1
    return ["playerID", "yearID", "teamID", "G"] + cols[start:end]


@pytest.fixture(scope="module")
def players_base(batting, sel_cols):
    return batting[sel_cols].order_by(sel_cols[:3])


@pytest.fixture(scope="module")
def players(players_base):
    return players_base.group_by("playerID")


@pytest.fixture(scope="module")
def players_df(players_base):
    return players_base.execute().reset_index(drop=True)
