from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ibis
from ibis import util
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.backends.tests.data import json_types, win


def set_pyspark_database(con, database):
    con._session.catalog.setCurrentDatabase(database)


class TestConf(BackendTest, RoundAwayFromZero):
    supported_to_timestamp_units = {"s"}
    deps = ("pyspark",)

    def _load_data(self, **_: Any) -> None:
        from pyspark.sql import Row

        s = self.connection._session
        num_partitions = 4

        sort_cols = {"functional_alltypes": "id"}

        for name in TEST_TABLES:
            path = str(self.data_dir / "parquet" / f"{name}.parquet")
            t = s.read.parquet(path).repartition(num_partitions)
            if (sort_col := sort_cols.get(name)) is not None:
                t = t.sort(sort_col)
            t.createOrReplaceTempView(name)

        s.createDataFrame([(1, "a")], ["foo", "bar"]).createOrReplaceTempView("simple")

        s.createDataFrame(
            [
                Row(abc=Row(a=1.0, b="banana", c=2)),
                Row(abc=Row(a=2.0, b="apple", c=3)),
                Row(abc=Row(a=3.0, b="orange", c=4)),
                Row(abc=Row(a=None, b="banana", c=2)),
                Row(abc=Row(a=2.0, b=None, c=3)),
                Row(abc=None),
                Row(abc=Row(a=3.0, b="orange", c=None)),
            ],
        ).createOrReplaceTempView("struct")

        s.createDataFrame(
            [([1, 2], [[3, 4], [5, 6]], {"a": [[2, 4], [3, 5]]})],
            [
                "list_of_ints",
                "list_of_list_of_ints",
                "map_string_list_of_list_of_ints",
            ],
        ).createOrReplaceTempView("nested_types")
        s.createDataFrame(
            [
                (
                    [1, 2, 3],
                    ["a", "b", "c"],
                    [1.0, 2.0, 3.0],
                    "a",
                    1.0,
                    [[], [1, 2, 3], None],
                ),
                ([4, 5], ["d", "e"], [4.0, 5.0], "a", 2.0, []),
                ([6, None], ["f", None], [6.0, None], "a", 3.0, [None, [], None]),
                (
                    [None, 1, None],
                    [None, "a", None],
                    [],
                    "b",
                    4.0,
                    [[1], [2], [], [3, 4, 5]],
                ),
                ([2, None, 3], ["b", None, "c"], None, "b", 5.0, None),
                (
                    [4, None, None, 5],
                    ["d", None, None, "e"],
                    [4.0, None, None, 5.0],
                    "c",
                    6.0,
                    [[1, 2, 3]],
                ),
            ],
            ["x", "y", "z", "grouper", "scalar_column", "multi_dim"],
        ).createOrReplaceTempView("array_types")

        s.createDataFrame(
            [({(1, 3): [[2, 4], [3, 5]]},)], ["map_tuple_list_of_list_of_ints"]
        ).createOrReplaceTempView("complicated")

        s.createDataFrame(
            [("a", 1, 4.0, "a"), ("b", 2, 5.0, "a"), ("c", 3, 6.0, "b")],
            ["a", "b", "c", "key"],
        ).createOrReplaceTempView("udf")

        s.createDataFrame(
            pd.DataFrame(
                {
                    "a": np.arange(10, dtype=float),
                    "b": [3.0, np.NaN] * 5,
                    "key": list("ddeefffggh"),
                }
            )
        ).createOrReplaceTempView("udf_nan")

        s.createDataFrame(
            [(float(i), None if i % 2 else 3.0, "ddeefffggh"[i]) for i in range(10)],
            ["a", "b", "key"],
        ).createOrReplaceTempView("udf_null")

        s.createDataFrame(
            pd.DataFrame(
                {
                    "a": np.arange(4.0).tolist() + np.random.rand(3).tolist(),
                    "b": np.arange(4.0).tolist() + np.random.rand(3).tolist(),
                    "key": list("ddeefff"),
                }
            )
        ).createOrReplaceTempView("udf_random")

        s.createDataFrame(json_types).createOrReplaceTempView("json_t")
        s.createDataFrame(win).createOrReplaceTempView("win")

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("ibis_testing")
            .master("local[1]")
            .config("spark.cores.max", 1)
            .config("spark.executor.heartbeatInterval", "3600s")
            .config("spark.executor.instances", 1)
            .config("spark.network.timeout", "4200s")
            .config("spark.sql.execution.arrow.pyspark.enabled", False)
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            .config("spark.storage.blockManagerSlaveTimeoutMs", "4200s")
            .config("spark.ui.showConsoleProgress", False)
            .config("spark.default.parallelism", 1)
            .config("spark.dynamicAllocation.enabled", False)
            .config("spark.rdd.compress", False)
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.shuffle.compress", False)
            .config("spark.shuffle.spill.compress", False)
            .config("spark.sql.shuffle.partitions", 1)
            .config("spark.ui.enabled", False)
            .getOrCreate()
        )
        return ibis.pyspark.connect(spark, **kw)


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    import pyspark.sql.functions as F
    import pyspark.sql.types as pt

    backend_test = TestConf.load_data(data_dir, tmp_path_factory, worker_id)
    con = backend_test.connection

    df = con._session.range(0, 10)
    df = df.withColumn("str_col", F.lit("value"))
    df.createTempView("basic_table")

    df_nulls = con._session.createDataFrame(
        [
            ["k1", np.NaN, "Alfred", None],
            ["k1", 3.0, None, "joker"],
            ["k2", 27.0, "Batman", "batmobile"],
            ["k2", None, "Catwoman", "motorcycle"],
        ],
        ["key", "age", "user", "toy"],
    )
    df_nulls.createTempView("null_table")

    df_dates = con._session.createDataFrame(
        [["2018-01-02"], ["2018-01-03"], ["2018-01-04"]], ["date_str"]
    )
    df_dates.createTempView("date_table")

    df_arrays = con._session.createDataFrame(
        [
            ["k1", [1, 2, 3], ["a"]],
            ["k2", [4, 5], ["test1", "test2", "test3"]],
            ["k3", [6], ["w", "x", "y", "z"]],
            ["k1", [], ["cat", "dog"]],
            ["k1", [7, 8], []],
        ],
        ["key", "array_int", "array_str"],
    )
    df_arrays.createTempView("array_table")

    df_time_indexed = con._session.createDataFrame(
        [
            [datetime(2017, 1, 2, 5, tzinfo=timezone.utc), 1, 1.0],
            [datetime(2017, 1, 2, 5, tzinfo=timezone.utc), 2, 2.0],
            [datetime(2017, 1, 2, 6, tzinfo=timezone.utc), 1, 3.0],
            [datetime(2017, 1, 2, 6, tzinfo=timezone.utc), 2, 4.0],
            [datetime(2017, 1, 2, 7, tzinfo=timezone.utc), 1, 5.0],
            [datetime(2017, 1, 2, 7, tzinfo=timezone.utc), 2, 6.0],
            [datetime(2017, 1, 4, 8, tzinfo=timezone.utc), 1, 7.0],
            [datetime(2017, 1, 4, 8, tzinfo=timezone.utc), 2, 8.0],
        ],
        ["time", "key", "value"],
    )

    df_time_indexed.createTempView("time_indexed_table")

    df_interval = con._session.createDataFrame(
        [
            [
                timedelta(days=10),
                timedelta(hours=10),
                timedelta(minutes=10),
                timedelta(seconds=10),
            ]
        ],
        pt.StructType(
            [
                pt.StructField(
                    "interval_day",
                    pt.DayTimeIntervalType(
                        pt.DayTimeIntervalType.DAY, pt.DayTimeIntervalType.DAY
                    ),
                ),
                pt.StructField(
                    "interval_hour",
                    pt.DayTimeIntervalType(
                        pt.DayTimeIntervalType.HOUR, pt.DayTimeIntervalType.HOUR
                    ),
                ),
                pt.StructField(
                    "interval_minute",
                    pt.DayTimeIntervalType(
                        pt.DayTimeIntervalType.MINUTE, pt.DayTimeIntervalType.MINUTE
                    ),
                ),
                pt.StructField(
                    "interval_second",
                    pt.DayTimeIntervalType(
                        pt.DayTimeIntervalType.SECOND, pt.DayTimeIntervalType.SECOND
                    ),
                ),
            ]
        ),
    )

    df_interval.createTempView("interval_table")

    df_interval_invalid = con._session.createDataFrame(
        [[timedelta(days=10, hours=10, minutes=10, seconds=10)]],
        pt.StructType(
            [
                pt.StructField(
                    "interval_day_hour",
                    pt.DayTimeIntervalType(
                        pt.DayTimeIntervalType.DAY, pt.DayTimeIntervalType.HOUR
                    ),
                )
            ]
        ),
    )

    df_interval_invalid.createTempView("invalid_interval_table")

    return con


class IbisWindow:
    # Test util class to generate different types of ibis windows
    def __init__(self, windows):
        self.windows = windows

    def get_windows(self):
        # Return a list of Ibis windows
        return [
            ibis.window(
                preceding=w[0],
                following=w[1],
                order_by="time",
                group_by="key",
            )
            for w in self.windows
        ]


@pytest.fixture
def ibis_windows(request):
    return IbisWindow(request.param).get_windows()


@pytest.fixture(scope="session", autouse=True)
def test_data_db(con):
    name = os.environ.get("IBIS_TEST_DATA_DB", "ibis_testing")
    con.create_database(name, force=True)
    set_pyspark_database(con, name)
    yield name
    con.drop_database(name, force=True)


@pytest.fixture
def temp_database(con, test_data_db):
    name = util.gen_name("database")
    con.create_database(name)
    yield name
    set_pyspark_database(con, test_data_db)
    con.drop_database(name, force=True)


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture
def temp_table_db(con, temp_database):
    name = util.gen_name("table")
    yield temp_database, name
    assert name in con.list_tables(database=temp_database), name
    con.drop_table(name, database=temp_database)
