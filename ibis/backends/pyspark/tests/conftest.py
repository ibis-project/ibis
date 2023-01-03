from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import ibis
from ibis import util
from ibis.backends.conftest import TEST_TABLES, _random_identifier
from ibis.backends.pyspark.datatypes import spark_dtype
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero
from ibis.backends.tests.data import win

pytest.importorskip("pyspark")

import pyspark.sql.functions as F  # noqa: E402
import pyspark.sql.types as pt  # noqa: E402
from pyspark.sql import Row, SparkSession  # noqa: E402


def get_common_spark_testing_client(data_directory, connect):
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
        .config('spark.default.parallelism', 1)
        .config('spark.dynamicAllocation.enabled', False)
        .config('spark.rdd.compress', False)
        .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
        .config('spark.shuffle.compress', False)
        .config('spark.shuffle.spill.compress', False)
        .config('spark.sql.shuffle.partitions', 1)
        .config('spark.ui.enabled', False)
        .getOrCreate()
    )
    _spark_testing_client = connect(spark)
    s = _spark_testing_client._session
    num_partitions = 4

    s.read.csv(
        path=str(data_directory / 'functional_alltypes.csv'),
        schema=spark_dtype(
            ibis.schema(
                {
                    # cast below, Spark can't read 0/1 as bool
                    name: {"bool_col": "int8"}.get(name, dtype)
                    for name, dtype in TEST_TABLES["functional_alltypes"].items()
                }
            )
        ),
        mode='FAILFAST',
        header=True,
    ).repartition(num_partitions).sort('index').withColumn(
        "bool_col", F.column("bool_col").cast("boolean")
    ).createOrReplaceTempView(
        'functional_alltypes'
    )

    for name, schema in TEST_TABLES.items():
        if name != "functional_alltypes":
            s.read.csv(
                path=str(data_directory / f'{name}.csv'),
                schema=spark_dtype(schema),
                header=True,
            ).repartition(num_partitions).createOrReplaceTempView(name)

    df_simple = s.createDataFrame([(1, 'a')], ['foo', 'bar'])
    df_simple.createOrReplaceTempView('simple')

    df_struct = s.createDataFrame(
        [
            Row(abc=Row(a=1.0, b='banana', c=2)),
            Row(abc=Row(a=2.0, b='apple', c=3)),
            Row(abc=Row(a=3.0, b='orange', c=4)),
            Row(abc=Row(a=None, b='banana', c=2)),
            Row(abc=Row(a=2.0, b=None, c=3)),
            Row(abc=None),
            Row(abc=Row(a=3.0, b='orange', c=None)),
        ],
        schema=pt.StructType(
            [
                pt.StructField(
                    "abc",
                    pt.StructType(
                        [
                            pt.StructField("a", pt.DoubleType(), True),
                            pt.StructField("b", pt.StringType(), True),
                            pt.StructField("c", pt.IntegerType(), True),
                        ]
                    ),
                )
            ]
        ),
    )
    df_struct.createOrReplaceTempView('struct')

    df_nested_types = s.createDataFrame(
        [([1, 2], [[3, 4], [5, 6]], {'a': [[2, 4], [3, 5]]})],
        [
            'list_of_ints',
            'list_of_list_of_ints',
            'map_string_list_of_list_of_ints',
        ],
    )
    df_nested_types.createOrReplaceTempView('nested_types')
    df_array_types = s.createDataFrame(
        [
            (
                [1, 2, 3],
                ['a', 'b', 'c'],
                [1.0, 2.0, 3.0],
                'a',
                1.0,
                [[], [1, 2, 3], None],
            ),
            ([4, 5], ['d', 'e'], [4.0, 5.0], 'a', 2.0, []),
            ([6, None], ['f', None], [6.0, None], 'a', 3.0, [None, [], None]),
            (
                [None, 1, None],
                [None, 'a', None],
                [],
                'b',
                4.0,
                [[1], [2], [], [3, 4, 5]],
            ),
            ([2, None, 3], ['b', None, 'c'], None, 'b', 5.0, None),
            (
                [4, None, None, 5],
                ['d', None, None, 'e'],
                [4.0, None, None, 5.0],
                'c',
                6.0,
                [[1, 2, 3]],
            ),
        ],
        ["x", "y", "z", "grouper", "scalar_column", "multi_dim"],
    )
    df_array_types.createOrReplaceTempView("array_types")

    df_complicated = s.createDataFrame(
        [({(1, 3): [[2, 4], [3, 5]]},)], ['map_tuple_list_of_list_of_ints']
    )
    df_complicated.createOrReplaceTempView('complicated')

    df_udf = s.createDataFrame(
        [('a', 1, 4.0, 'a'), ('b', 2, 5.0, 'a'), ('c', 3, 6.0, 'b')],
        ['a', 'b', 'c', 'key'],
    )
    df_udf.createOrReplaceTempView('udf')

    df_udf_nan = s.createDataFrame(
        pd.DataFrame(
            {
                'a': np.arange(10, dtype=float),
                'b': [3.0, np.NaN] * 5,
                'key': list('ddeefffggh'),
            }
        )
    )
    df_udf_nan.createOrReplaceTempView('udf_nan')

    df_udf_null = s.createDataFrame(
        [(float(i), None if i % 2 else 3.0, 'ddeefffggh'[i]) for i in range(10)],
        ['a', 'b', 'key'],
    )
    df_udf_null.createOrReplaceTempView('udf_null')

    df_udf_random = s.createDataFrame(
        pd.DataFrame(
            {
                'a': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
                'b': np.arange(4, dtype=float).tolist() + np.random.rand(3).tolist(),
                'key': list('ddeefff'),
            }
        )
    )
    df_udf_random.createOrReplaceTempView('udf_random')

    df_json_t = s.createDataFrame(
        pd.DataFrame(
            {
                "js": [
                    '{"a": [1,2,3,4], "b": 1}',
                    '{"a":null,"b":2}',
                    '{"a":"foo", "c":null}',
                    "null",
                    "[42,47,55]",
                    "[]",
                ]
            }
        )
    )
    df_json_t.createOrReplaceTempView("json_t")

    win_t = s.createDataFrame(win)
    win_t.createOrReplaceTempView("win")

    return _spark_testing_client


def get_pyspark_testing_client(data_directory):
    return get_common_spark_testing_client(data_directory, ibis.pyspark.connect)


class TestConf(BackendTest, RoundAwayFromZero):
    supported_to_timestamp_units = {'s'}

    @staticmethod
    def connect(data_directory):
        return get_pyspark_testing_client(data_directory)


@pytest.fixture(scope='session')
def client(data_directory):
    client = get_pyspark_testing_client(data_directory)

    df = client._session.range(0, 10)
    df = df.withColumn("str_col", F.lit('value'))
    df.createTempView('basic_table')

    df_nulls = client._session.createDataFrame(
        [
            ['k1', np.NaN, 'Alfred', None],
            ['k1', 3.0, None, 'joker'],
            ['k2', 27.0, 'Batman', 'batmobile'],
            ['k2', None, 'Catwoman', 'motorcycle'],
        ],
        ['key', 'age', 'user', 'toy'],
    )
    df_nulls.createTempView('null_table')

    df_dates = client._session.createDataFrame(
        [['2018-01-02'], ['2018-01-03'], ['2018-01-04']], ['date_str']
    )
    df_dates.createTempView('date_table')

    df_arrays = client._session.createDataFrame(
        [
            ['k1', [1, 2, 3], ['a']],
            ['k2', [4, 5], ['test1', 'test2', 'test3']],
            ['k3', [6], ['w', 'x', 'y', 'z']],
            ['k1', [], ['cat', 'dog']],
            ['k1', [7, 8], []],
        ],
        ['key', 'array_int', 'array_str'],
    )
    df_arrays.createTempView('array_table')

    df_time_indexed = client._session.createDataFrame(
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
        ['time', 'key', 'value'],
    )

    df_time_indexed.createTempView('time_indexed_table')

    return client


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
                order_by='time',
                group_by='key',
            )
            for w in self.windows
        ]


@pytest.fixture
def ibis_windows(request):
    return IbisWindow(request.param).get_windows()


@pytest.fixture(scope='session', autouse=True)
def test_data_db(client):
    name = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')
    client.create_database(name)
    try:
        client.set_database(name)
        yield name
    finally:
        client.drop_database(name, force=True)


@pytest.fixture
def temp_database(client, test_data_db):
    name = _random_identifier('database')
    client.create_database(name)
    try:
        yield name
    finally:
        client.set_database(test_data_db)
        client.drop_database(name, force=True)


@pytest.fixture
def temp_table(client):
    name = _random_identifier('table')
    try:
        yield name
    finally:
        assert name in client.list_tables(), name
        client.drop_table(name)


@pytest.fixture(scope='session')
def alltypes(client):
    return client.table('functional_alltypes').relabel({'Unnamed: 0': 'Unnamed:0'})


@pytest.fixture(scope='session')
def tmp_dir():
    return f'/tmp/__ibis_test_{util.guid()}'


@pytest.fixture
def temp_table_db(client, temp_database):
    name = _random_identifier('table')
    try:
        yield temp_database, name
    finally:
        assert name in client.list_tables(database=temp_database), name
        client.drop_table(name, database=temp_database)


@pytest.fixture
def temp_view(client):
    name = _random_identifier('view')
    try:
        yield name
    finally:
        assert name in client.list_tables(), name
        client.drop_view(name)
