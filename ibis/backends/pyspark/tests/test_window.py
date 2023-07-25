from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis

pyspark = pytest.importorskip("pyspark")

import pyspark.sql.functions as F  # noqa: E402
from pyspark.sql.window import Window  # noqa: E402


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
def test_time_indexed_window(con, ibis_windows, spark_range):
    table = con.table("time_indexed_table")
    result = table.mutate(mean=table["value"].mean().over(ibis_windows[0])).compile()
    result_pd = result.toPandas()
    spark_table = table.compile()
    spark_window = (
        Window.partitionBy("key")
        .orderBy(F.col("time").cast("long"))
        .rangeBetween(*spark_range)
    )
    expected = spark_table.withColumn(
        "mean",
        F.mean(spark_table["value"]).over(spark_window),
    ).toPandas()
    tm.assert_frame_equal(result_pd, expected)


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
def test_multiple_windows(con, ibis_windows, spark_range):
    table = con.table("time_indexed_table")
    result = table.mutate(
        mean_1h=table["value"].mean().over(ibis_windows[0]),
        mean_2h=table["value"].mean().over(ibis_windows[1]),
    ).compile()
    result_pd = result.toPandas()

    spark_table = table.compile()
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
    tm.assert_frame_equal(result_pd, expected)
