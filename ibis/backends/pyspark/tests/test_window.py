from __future__ import annotations

from time import sleep

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis import _

pyspark = pytest.importorskip("pyspark")

import pyspark.sql.functions as F  # noqa: E402
from pyspark.sql.window import Window  # noqa: E402


@pytest.fixture
def t(con):
    return con.table("time_indexed_table")


@pytest.fixture
def spark_table(con):
    return con._session.table("time_indexed_table")


@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [
        ([(ibis.interval(hours=1), 0)], (-3600, 0)),  # 1h back looking window
        ([(ibis.interval(hours=2), 0)], (-7200, 0)),  # 2h back looking window
        (
            [(0, ibis.interval(hours=1))],
            (0, 3600),
        ),  # 1h forward looking window
    ],
    indirect=["ibis_windows"],
)
def test_time_indexed_window(t, spark_table, ibis_windows, spark_range):
    result = t.mutate(mean=t["value"].mean().over(ibis_windows[0])).execute()

    spark_window = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range)
    )
    expected = spark_table.withColumn(
        "mean",
        F.mean(spark_table["value"]).over(spark_window),
    ).toPandas()

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("ibis_windows", "spark_range"),
    [
        (
            [(ibis.interval(hours=1), 0), (ibis.interval(hours=2), 0)],
            [(-3600, 0), (-7200, 0)],
        ),
    ],
    indirect=["ibis_windows"],
)
def test_multiple_windows(t, spark_table, ibis_windows, spark_range):
    result = t.mutate(
        mean_1h=t["value"].mean().over(ibis_windows[0]),
        mean_2h=t["value"].mean().over(ibis_windows[1]),
    ).execute()

    spark_window = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[0])
    )
    spark_window_2 = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range[1])
    )
    expected = (
        spark_table.withColumn(
            "mean_1h",
            F.mean(spark_table["value"]).over(spark_window),
        )
        .withColumn(
            "mean_2h",
            F.mean(spark_table["value"]).over(spark_window_2),
        )
        .toPandas()
    )
    tm.assert_frame_equal(result, expected)


def test_tumble_window_by_grouped_agg(con_streaming, tmp_path):
    t = con_streaming.table("functional_alltypes")
    expr = (
        t.window_by(t.timestamp_col)
        .tumble(size=ibis.interval(seconds=30))
        .agg(by=["string_col"], avg=_.float_col.mean())
    )
    path = tmp_path / "out"
    con_streaming.to_csv_dir(
        expr,
        path=path,
        options={"checkpointLocation": tmp_path / "checkpoint", "header": True},
    )
    sleep(5)
    dfs = [pd.read_csv(f) for f in path.glob("*.csv")]
    df = pd.concat([df for df in dfs if not df.empty])
    assert list(df.columns) == ["window_start", "window_end", "string_col", "avg"]
    # [NOTE] The expected number of rows here is 7299 because when all the data is ready
    # at once, no event is dropped as out of order. On the contrary, Flink discards all
    # out-of-order events as late arrivals and only emits 610 windows.
    assert df.shape == (7299, 4)


def test_tumble_window_by_ungrouped_agg(con_streaming, tmp_path):
    t = con_streaming.table("functional_alltypes")
    expr = (
        t.window_by(t.timestamp_col)
        .tumble(size=ibis.interval(seconds=30))
        .agg(avg=_.float_col.mean())
    )
    path = tmp_path / "out"
    con_streaming.to_csv_dir(
        expr,
        path=path,
        options={"checkpointLocation": tmp_path / "checkpoint", "header": True},
    )
    sleep(5)
    dfs = [pd.read_csv(f) for f in path.glob("*.csv")]
    df = pd.concat([df for df in dfs if not df.empty])
    assert list(df.columns) == ["window_start", "window_end", "avg"]
    assert df.shape == (7299, 3)
