from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
from ibis.common.exceptions import IbisTypeError

pyspark = pytest.importorskip("pyspark")


@pytest.fixture
def t(con):
    return con.table("basic_table")


@pytest.fixture
def df(con):
    return con._session.table("basic_table").toPandas()


def test_basic(t):
    result = t.execute()
    expected = pd.DataFrame({"id": range(10), "str_col": "value"})
    tm.assert_frame_equal(result, expected)


def test_projection(t):
    result1 = t.mutate(v=t["id"]).execute()
    expected1 = pd.DataFrame({"id": range(10), "str_col": "value", "v": range(10)})

    result2 = t.mutate(v=t["id"]).mutate(v2=t["id"]).mutate(id=t["id"] * 2).execute()
    expected2 = pd.DataFrame(
        {
            "id": range(0, 20, 2),
            "str_col": "value",
            "v": range(10),
            "v2": range(10),
        }
    )

    tm.assert_frame_equal(result1, expected1)
    tm.assert_frame_equal(result2, expected2)


def test_aggregation_col(t, df):
    result = t["id"].count().execute()
    assert result == len(df)


def test_aggregation(t, df):
    result = t.aggregate(max=t["id"].max()).execute()
    expected = pd.DataFrame({"max": [df.id.max()]})
    tm.assert_frame_equal(result, expected)


def test_group_by(t, df):
    result = t.group_by("id").aggregate(max=t["id"].max()).execute()
    expected = df[["id"]].assign(max=df.groupby("id").id.max())
    tm.assert_frame_equal(result, expected)


def test_window(t, df):
    w = ibis.window()
    result = t.mutate(grouped_demeaned=t["id"] - t["id"].mean().over(w)).execute()
    expected = df.assign(grouped_demeaned=df.id - df.id.mean())

    tm.assert_frame_equal(result, expected)


def test_greatest(t, df):
    result = t.mutate(greatest=ibis.greatest(t.id, t.id + 1)).execute()
    expected = df.assign(greatest=df.id + 1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("filter_fn", "expected_fn"),
    [
        param(lambda t: t.filter(t.id < 5), lambda df: df[df.id < 5]),
        param(lambda t: t.filter(t.id != 5), lambda df: df[df.id != 5]),
        param(
            lambda t: t.filter([t.id < 5, t.str_col == "na"]),
            lambda df: df[df.id < 5][df.str_col == "na"],
        ),
        param(
            lambda t: t.filter((t.id > 3) & (t.id < 11)),
            lambda df: df[(df.id > 3) & (df.id < 11)],
        ),
        param(
            lambda t: t.filter((t.id == 3) | (t.id == 5)),
            lambda df: df[(df.id == 3) | (df.id == 5)],
        ),
    ],
)
def test_filter(t, df, filter_fn, expected_fn):
    result = filter_fn(t).execute().reset_index(drop=True)
    expected = expected_fn(df).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_cast(t, df):
    result = t.mutate(id_string=t.id.cast("string")).execute()
    df = df.assign(id_string=df.id.astype(str))
    tm.assert_frame_equal(result, df)


def test_alias_after_select(t, df):
    # Regression test for issue 2136
    table = t[["id"]]
    table = table.mutate(id2=table["id"])
    result = table.execute()
    tm.assert_series_equal(result["id"], result["id2"], check_names=False)


def test_interval_columns_invalid(con):
    msg = r"DayTimeIntervalType\(0, 1\) couldn't be converted to Interval"
    with pytest.raises(IbisTypeError, match=msg):
        con.table("invalid_interval_table")


def test_string_literal_backslash_escaping(con):
    expr = ibis.literal("\\d\\e")
    result = con.execute(expr)
    assert result == "\\d\\e"
