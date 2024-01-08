from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis

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
