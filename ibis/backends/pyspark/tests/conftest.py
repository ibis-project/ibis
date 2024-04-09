from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import ibis
from ibis import util
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest
from ibis.backends.tests.data import json_types, topk, win

if TYPE_CHECKING:
    from pathlib import Path


def set_pyspark_database(con, database):
    con._session.catalog.setCurrentDatabase(database)


class TestConf(BackendTest):
    deps = ("pyspark",)

    def _load_data_helper(self, for_streaming: bool = False):
        from pyspark.sql import Row

        get_table_name = (
            (lambda name: f"{name}_streaming") if for_streaming else (lambda name: name)
        )

        s = self.connection._session
        num_partitions = 4
        sort_cols = {"functional_alltypes": "id"}

        if for_streaming:
            # "By default, Structured Streaming from file based sources
            # requires you to specify the schema, rather than rely on
            # Spark to infer it automatically. ... .  For ad-hoc use
            # cases, you can re-enable schema inference by setting
            # spark.sql.streaming.schemaInference to true."
            # Ref: https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#schema-inference-and-partition-of-streaming-dataframesdatasets
            s.sql("set spark.sql.streaming.schemaInference=true")

        for name in TEST_TABLES:
            path = str(self.data_dir / "parquet" / f"{name}.parquet")

            t = (
                s.readStream.format("parquet").parquet(path).repartition(num_partitions)
                if for_streaming
                else s.read.parquet(path).repartition(num_partitions)
            )
            if (sort_col := sort_cols.get(name)) is not None:
                t = t.sort(sort_col)
            t.createOrReplaceTempView(get_table_name(name))

        s.createDataFrame([(1, "a")], ["foo", "bar"]).createOrReplaceTempView(
            get_table_name("simple")
        )

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
        ).createOrReplaceTempView(get_table_name("struct"))

        s.createDataFrame(
            [([1, 2], [[3, 4], [5, 6]], {"a": [[2, 4], [3, 5]]})],
            [
                "list_of_ints",
                "list_of_list_of_ints",
                "map_string_list_of_list_of_ints",
            ],
        ).createOrReplaceTempView(get_table_name("nested_types"))
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
        ).createOrReplaceTempView(get_table_name("array_types"))

        s.createDataFrame(
            [({(1, 3): [[2, 4], [3, 5]]},)], ["map_tuple_list_of_list_of_ints"]
        ).createOrReplaceTempView(get_table_name("complicated"))

        s.createDataFrame(
            [("a", 1, 4.0, "a"), ("b", 2, 5.0, "a"), ("c", 3, 6.0, "b")],
            ["a", "b", "c", "key"],
        ).createOrReplaceTempView(get_table_name("udf"))

        s.createDataFrame(
            pd.DataFrame(
                {
                    "a": np.arange(10, dtype=float),
                    "b": [3.0, np.NaN] * 5,
                    "key": list("ddeefffggh"),
                }
            )
        ).createOrReplaceTempView(get_table_name("udf_nan"))

        s.createDataFrame(
            [(float(i), None if i % 2 else 3.0, "ddeefffggh"[i]) for i in range(10)],
            ["a", "b", "key"],
        ).createOrReplaceTempView(get_table_name("udf_null"))

        s.createDataFrame(
            pd.DataFrame(
                {
                    "a": np.arange(4.0).tolist() + np.random.rand(3).tolist(),
                    "b": np.arange(4.0).tolist() + np.random.rand(3).tolist(),
                    "key": list("ddeefff"),
                }
            )
        ).createOrReplaceTempView(get_table_name("udf_random"))

        s.createDataFrame(json_types).createOrReplaceTempView(get_table_name("json_t"))
        s.createDataFrame(win).createOrReplaceTempView(get_table_name("win"))
        s.createDataFrame(topk.to_pandas()).createOrReplaceTempView(
            get_table_name("topk")
        )

    def _load_data(self, **kw: Any) -> None:
        self._load_data_helper()

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        # Spark internally stores timestamps as UTC values, and timestamp
        # data that is brought in without a specified time zone is
        # converted as local time to UTC with microsecond resolution.
        # https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics

        from pyspark.sql import SparkSession

        config = (
            SparkSession.builder.appName("ibis_testing")
            .master("local[1]")
            .config("spark.cores.max", 1)
            .config("spark.default.parallelism", 1)
            .config("spark.driver.extraJavaOptions", "-Duser.timezone=GMT")
            .config("spark.dynamicAllocation.enabled", False)
            .config("spark.executor.extraJavaOptions", "-Duser.timezone=GMT")
            .config("spark.executor.heartbeatInterval", "3600s")
            .config("spark.executor.instances", 1)
            .config("spark.network.timeout", "4200s")
            .config("spark.rdd.compress", False)
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.shuffle.compress", False)
            .config("spark.shuffle.spill.compress", False)
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", 1)
            .config("spark.storage.blockManagerSlaveTimeoutMs", "4200s")
            .config("spark.ui.enabled", False)
            .config("spark.ui.showConsoleProgress", False)
            .config("spark.sql.execution.arrow.pyspark.enabled", False)
        )

        try:
            from delta.pip_utils import configure_spark_with_delta_pip
        except ImportError:
            configure_spark_with_delta_pip = lambda cfg: cfg
        else:
            config = config.config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            ).config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")

        spark = configure_spark_with_delta_pip(config).getOrCreate()
        return ibis.pyspark.connect(spark, **kw)


class TestConfForStreaming(TestConf):
    # Note: The same session can be used for both batch and streaming
    # jobs in Spark. Streaming is made explicit on the source
    # dataframes. This is why, we do not really need a separate
    # `TestConf` class for streaming, but only need to create
    # streaming counterparts for the test tables. Still added this
    # class to keep the testing uniform with Flink. This class is used
    # in `con_streaming()` fixture (streaming counterpart of `con()`)
    # to create a new spark session and load the `***_streaming`
    # tables for testing.  However, either `con()` or
    # `con_streaming()` can be used to execute any batch/streaming
    # job. This is why, we set `autouse=True` for `con_streaming()`
    # to create the streaming tables, and then rely solely on `con()`
    # to operate on those tables in the tests.

    @classmethod
    def name(cls) -> str:
        # To pass `if not fn.exists()` in `BackendTest.stateful_load()`.
        return f"{cls.__bases__[0].name()}_streaming"

    def _load_data(self, **_: Any) -> None:
        self._load_data_helper(for_streaming=True)


def create_connection(
    data_dir: Path,
    tmpdir: Path,
    worker_id: str,
    for_streaming: bool = False,
):
    import tempfile

    import pyspark.sql.types as pt

    from ibis.backends.pyspark.tests import utils

    if for_streaming:
        backend_test = TestConfForStreaming.load_data(data_dir, tmpdir, worker_id)
    else:
        backend_test = TestConf.load_data(data_dir, tmpdir, worker_id)
    con = backend_test.connection

    tmpdir = tempfile.mkdtemp()

    def create_temp_view_for_streaming(
        table_name: str,
        df: pd.DataFrame | None = None,
        rows: list[str] | None = None,
        columns: list[str] | None = None,
        schema: pt.DataType | None = None,
    ):
        if for_streaming:
            path = f"{tmpdir}/{table_name}.parquet"
            if df is None:
                df = pd.DataFrame(rows, columns=columns)
            df.to_parquet(path)

            if schema is None:
                schema = utils.spark_schema_from_df(df)

            (
                con._session.readStream.schema(schema)
                .format("parquet")
                .parquet(f"{tmpdir}/{table_name}*.parquet")
                .createTempView(f"{table_name}_streaming")
            )

        else:
            con._session.createDataFrame(rows, schema or columns).createTempView(
                table_name
            )

    create_temp_view_for_streaming(
        table_name="basic_table",
        # df=pd.DataFrame({"str_col": list(range(10))}),
        rows=[[i, "value"] for i in range(10)],
        columns=["id", "str_col"],
        schema=pt.StructType(
            [
                pt.StructField("id", pt.LongType()),
                pt.StructField("str_col", pt.StringType()),
            ]
        ),
    )

    create_temp_view_for_streaming(
        table_name="null_table",
        rows=[
            ["k1", np.NaN, "Alfred", None],
            ["k1", 3.0, None, "joker"],
            ["k2", 27.0, "Batman", "batmobile"],
            ["k2", None, "Catwoman", "motorcycle"],
        ],
        columns=["key", "age", "user", "toy"],
        schema=pt.StructType(
            [
                pt.StructField("key", pt.StringType()),
                pt.StructField("age", pt.DoubleType()),
                pt.StructField("user", pt.StringType()),
                pt.StructField("toy", pt.StringType()),
            ]
        ),
    )

    create_temp_view_for_streaming(
        table_name="date_table",
        rows=[["2018-01-02"], ["2018-01-03"], ["2018-01-04"]],
        columns=["date_str"],
        schema=pt.StructType([pt.StructField("date_str", pt.StringType())]),
    )

    create_temp_view_for_streaming(
        table_name="array_table",
        rows=[
            ["k1", [1, 2, 3], ["a"]],
            ["k2", [4, 5], ["test1", "test2", "test3"]],
            ["k3", [6], ["w", "x", "y", "z"]],
            ["k1", [], ["cat", "dog"]],
            ["k1", [7, 8], []],
        ],
        columns=["key", "array_int", "array_str"],
        schema=pt.StructType(
            [
                pt.StructField("key", pt.StringType()),
                pt.StructField("array_int", pt.ArrayType(pt.LongType())),
                pt.StructField("array_str", pt.ArrayType(pt.StringType())),
            ]
        ),
    )

    create_temp_view_for_streaming(
        table_name="time_indexed_table",
        rows=[
            [datetime(2017, 1, 2, 5, tzinfo=timezone.utc), 1, 1.0],
            [datetime(2017, 1, 2, 5, tzinfo=timezone.utc), 2, 2.0],
            [datetime(2017, 1, 2, 6, tzinfo=timezone.utc), 1, 3.0],
            [datetime(2017, 1, 2, 6, tzinfo=timezone.utc), 2, 4.0],
            [datetime(2017, 1, 2, 7, tzinfo=timezone.utc), 1, 5.0],
            [datetime(2017, 1, 2, 7, tzinfo=timezone.utc), 2, 6.0],
            [datetime(2017, 1, 4, 8, tzinfo=timezone.utc), 1, 7.0],
            [datetime(2017, 1, 4, 8, tzinfo=timezone.utc), 2, 8.0],
        ],
        columns=["time", "key", "value"],
        schema=pt.StructType(
            [
                pt.StructField("time", pt.TimestampType()),
                pt.StructField("key", pt.IntegerType()),
                pt.StructField("value", pt.FloatType()),
            ]
        ),
    )

    create_temp_view_for_streaming(
        table_name="interval_table",
        rows=[
            [
                timedelta(days=10),
                timedelta(hours=10),
                timedelta(minutes=10),
                timedelta(seconds=10),
            ]
        ],
        columns=["interval_day", "interval_hour", "interval_minute", "interval_second"],
        schema=pt.StructType(
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

    create_temp_view_for_streaming(
        table_name="invalid_interval_table",
        rows=[[timedelta(days=10, hours=10, minutes=10, seconds=10)]],
        columns=["interval_day_hour"],
        schema=pt.StructType(
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

    return con


@pytest.fixture(scope="session")
def con(data_dir, tmp_path_factory, worker_id):
    return create_connection(
        data_dir=data_dir, tmpdir=tmp_path_factory, worker_id=worker_id
    )


@pytest.fixture(scope="session", autouse=True)
def con_streaming(data_dir, tmp_path_factory, worker_id):
    """Fixture for creating the `***_streaming` tables, using which we
    run the tests with pyspark-streaming. For more context on this
    fixture, refer to the note added under `TestConfForStreaming`.
    """
    return create_connection(
        data_dir=data_dir,
        tmpdir=tmp_path_factory,
        worker_id=worker_id,
        for_streaming=True,
    )


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


@pytest.fixture(scope="session")
def alltypes_streaming(con_streaming):
    return con_streaming.table("functional_alltypes_streaming")


@pytest.fixture
def temp_table_db(con, temp_database):
    name = util.gen_name("table")
    yield temp_database, name
    assert name in con.list_tables(database=temp_database), name

    con.drop_table(name, database=temp_database, force=True)
