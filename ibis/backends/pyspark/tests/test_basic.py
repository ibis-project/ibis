from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
from ibis.common.exceptions import IbisTypeError
from ibis.expr import datatypes as dt

pyspark = pytest.importorskip("pyspark")

import pyspark.sql.functions as F  # noqa: E402

from ibis.backends.pyspark.compiler import _can_be_replaced_by_column_name  # noqa: E402


def test_basic(con):
    table = con.table("basic_table")
    result = table.compile().toPandas()
    expected = pd.DataFrame({"id": range(10), "str_col": "value"})

    tm.assert_frame_equal(result, expected)


def test_projection(con):
    table = con.table("basic_table")
    result1 = table.mutate(v=table["id"]).compile().toPandas()

    expected1 = pd.DataFrame({"id": range(10), "str_col": "value", "v": range(10)})

    result2 = (
        table.mutate(v=table["id"])
        .mutate(v2=table["id"])
        .mutate(id=table["id"] * 2)
        .compile()
        .toPandas()
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


def test_aggregation_col(con):
    table = con.table("basic_table")
    result = table["id"].count().execute()
    assert result == table.compile().count()


def test_aggregation(con):
    table = con.table("basic_table")
    result = table.aggregate(max=table["id"].max()).compile()
    expected = table.compile().agg(F.max("id").alias("max"))

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_group_by(con):
    table = con.table("basic_table")
    result = table.group_by("id").aggregate(max=table["id"].max()).compile()
    expected = table.compile().groupby("id").agg(F.max("id").alias("max"))

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_window(con):
    table = con.table("basic_table")
    w = ibis.window()
    result = table.mutate(
        grouped_demeaned=table["id"] - table["id"].mean().over(w)
    ).compile()

    spark_window = pyspark.sql.Window.partitionBy()
    spark_table = table.compile()
    expected = spark_table.withColumn(
        "grouped_demeaned",
        spark_table["id"] - F.mean(spark_table["id"]).over(spark_window),
    )

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_greatest(con):
    table = con.table("basic_table")
    result = table.mutate(greatest=ibis.greatest(table.id)).compile()
    df = table.compile()
    expected = table.compile().withColumn("greatest", df.id)

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_selection(con):
    table = con.table("basic_table")
    table = table.mutate(id2=table["id"] * 2)

    result1 = table[["id"]].compile()
    result2 = table[["id", "id2"]].compile()
    result3 = table[[table, (table.id + 1).name("plus1")]].compile()
    result4 = table[[(table.id + 1).name("plus1"), table]].compile()

    df = table.compile()
    tm.assert_frame_equal(result1.toPandas(), df[["id"]].toPandas())
    tm.assert_frame_equal(result2.toPandas(), df[["id", "id2"]].toPandas())
    tm.assert_frame_equal(
        result3.toPandas(),
        df[[df.columns]].withColumn("plus1", df.id + 1).toPandas(),
    )
    tm.assert_frame_equal(
        result4.toPandas(),
        df.withColumn("plus1", df.id + 1)[["plus1", *df.columns]].toPandas(),
    )


def test_join(con):
    table = con.table("basic_table")
    result = table.join(table, ["id", "str_col"])[table.id, table.str_col].compile()
    spark_table = table.compile()
    expected = spark_table.join(spark_table, ["id", "str_col"])

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


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
def test_filter(con, filter_fn, expected_fn):
    table = con.table("basic_table")

    result = filter_fn(table).compile()

    df = table.compile()
    expected = expected_fn(df)

    tm.assert_frame_equal(result.toPandas(), expected.toPandas())


def test_cast(con):
    table = con.table("basic_table")

    result = table.mutate(id_string=table.id.cast("string")).compile()

    df = table.compile()
    df = df.withColumn("id_string", df.id.cast("string"))

    tm.assert_frame_equal(result.toPandas(), df.toPandas())


def test_alias_after_select(con):
    # Regression test for issue 2136
    table = con.table("basic_table")
    table = table[["id"]]
    table = table.mutate(id2=table["id"])

    result = table.compile().toPandas()
    tm.assert_series_equal(result["id"], result["id2"], check_names=False)


@pytest.mark.parametrize(
    ("selection_fn", "selection_idx", "expected"),
    [
        # selected column id is selections[0], OK to replace since
        # id == t['id'] (straightforward column projection)
        (lambda t: t[["id"]], 0, True),
        # new column v is selections[1], cannot be replaced since it does
        # not exist in the root table
        (lambda t: t.mutate(v=t["id"]), 1, False),
        # new column id is selections[0], cannot be replaced since
        # new id != t['id']
        (lambda t: t.mutate(id=t["str_col"]), 0, False),
        # new column id is selections[0], OK to replace since
        # new id == t['id'] (mutation is no-op)
        (lambda t: t.mutate(id=t["id"]), 0, True),
        # new column id is selections[0], cannot be replaced since
        # new id != t['id']
        (lambda t: t.mutate(id=t["id"] + 1), 0, False),
        # new column id is selections[0], OK to replace since
        # new id == t['id'] (rename is a no-op)
        (lambda t: t.rename({"id": "id"}), 0, True),
        # new column id2 is selections[0], cannot be replaced since
        # id2 does not exist in the table
        (lambda t: t.rename({"id2": "id"}), 0, False),
    ],
)
def test_can_be_replaced_by_column_name(selection_fn, selection_idx, expected):
    table = ibis.table([("id", "double"), ("str_col", "string")])
    table = selection_fn(table)
    selection_to_test = table.op().selections[selection_idx]
    result = _can_be_replaced_by_column_name(selection_to_test, table.op().table)
    assert result == expected


def test_interval_columns(con):
    table = con.table("interval_table")
    assert table.schema() == ibis.schema(
        pairs=[
            ("interval_day", dt.Interval("D")),
            ("interval_hour", dt.Interval("h")),
            ("interval_minute", dt.Interval("m")),
            ("interval_second", dt.Interval("s")),
        ]
    )

    expected = pd.DataFrame(
        {
            "interval_day": [pd.Timedelta("10d")],
            "interval_hour": [pd.Timedelta("10h")],
            "interval_minute": [pd.Timedelta("10m")],
            "interval_second": [pd.Timedelta("10s")],
        }
    )
    tm.assert_frame_equal(table.execute(), expected)


def test_interval_columns_invalid(con):
    msg = r"DayTimeIntervalType\(0, 1\) couldn't be converted to Interval"
    with pytest.raises(IbisTypeError, match=msg):
        con.table("invalid_interval_table")
