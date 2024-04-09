from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
from ibis.backends.tests.errors import PySparkAnalysisException
from ibis.common.exceptions import IbisTypeError

pyspark = pytest.importorskip("pyspark")


@pytest.fixture(scope="session")
def batch_table(con):
    return con.table("basic_table")


@pytest.fixture(scope="session")
def streaming_table(con_streaming):
    return con_streaming.table("basic_table_streaming")


@pytest.fixture(params=["batch_table", "streaming_table"])
def table(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def df(con):
    return con._session.table("basic_table").toPandas()


def test_basic(table):
    result = table.execute()
    expected = pd.DataFrame({"id": range(10), "str_col": "value"})
    tm.assert_frame_equal(result, expected)


def test_projection(table):
    result1 = table.mutate(v=table["id"]).execute()
    expected1 = pd.DataFrame({"id": range(10), "str_col": "value", "v": range(10)})

    result2 = (
        table.mutate(v=table["id"])
        .mutate(v2=table["id"])
        .mutate(id=table["id"] * 2)
        .execute()
    )
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


@pytest.mark.parametrize(
    "table_fixture",
    [
        param("batch_table", id="batch"),
        param(
            "streaming_table",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Streaming aggregations require watermark.",
            ),
            id="streaming",
        ),
    ],
)
def test_aggregation_col(table_fixture, df, request):
    table = request.getfixturevalue(table_fixture)

    result = table["id"].count().execute()
    assert result == len(df)


@pytest.mark.parametrize(
    "table_fixture",
    [
        param("batch_table", id="batch"),
        param(
            "streaming_table",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Streaming aggregations require watermark.",
            ),
            id="streaming",
        ),
    ],
)
def test_aggregation(table_fixture, df, request):
    table = request.getfixturevalue(table_fixture)

    result = table.aggregate(max=table["id"].max()).execute()
    expected = pd.DataFrame({"max": [df.id.max()]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "table_fixture",
    [
        param("batch_table", id="batch"),
        param(
            "streaming_table",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Streaming aggregations require watermark.",
            ),
            id="streaming",
        ),
    ],
)
def test_group_by(table_fixture, df, request):
    table = request.getfixturevalue(table_fixture)

    result = table.group_by("id").aggregate(max=table["id"].max()).execute()
    expected = df[["id"]].assign(max=df.groupby("id").id.max())
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "table_fixture",
    [
        param("batch_table", id="batch"),
        param(
            "streaming_table",
            marks=pytest.mark.xfail(
                raises=PySparkAnalysisException,
                reason="Only time-window aggregations are supported in streaming.",
            ),
            id="streaming",
        ),
    ],
)
def test_window(table_fixture, df, request):
    table = request.getfixturevalue(table_fixture)

    w = ibis.window()
    result = table.mutate(
        grouped_demeaned=table["id"] - table["id"].mean().over(w)
    ).execute()
    expected = df.assign(grouped_demeaned=df.id - df.id.mean())

    tm.assert_frame_equal(result, expected)


def test_greatest(table, df):
    result = table.mutate(greatest=ibis.greatest(table.id, table.id + 1)).execute()
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
def test_filter(table, df, filter_fn, expected_fn):
    result = filter_fn(table).execute().reset_index(drop=True)
    expected = expected_fn(df).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_cast(table, df):
    result = table.mutate(id_string=table.id.cast("string")).execute()
    df = df.assign(id_string=df.id.astype(str))
    tm.assert_frame_equal(result, df)


def test_alias_after_select(table, df):
    # Regression test for issue 2136
    table = table[["id"]]
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


def test_connect_without_explicit_session():
    con = ibis.pyspark.connect()
    result = con.sql("SELECT CAST(1 AS BIGINT) as foo").to_pandas()
    tm.assert_frame_equal(result, pd.DataFrame({"foo": [1]}))
