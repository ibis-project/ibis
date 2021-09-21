import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as pt
import pytest
from pyspark.sql import SparkSession

import ibis
from ibis import util
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero

_pyspark_testing_client = None


def get_common_spark_testing_client(data_directory, connect):
    spark = (
        SparkSession.builder.config('spark.default.parallelism', 4)
        .config('spark.driver.bindAddress', '127.0.0.1')
        .getOrCreate()
    )
    _spark_testing_client = connect(spark)
    s = _spark_testing_client._session
    num_partitions = 4

    df_functional_alltypes = (
        s.read.csv(
            path=str(data_directory / 'functional_alltypes.csv'),
            schema=pt.StructType(
                [
                    pt.StructField('index', pt.IntegerType(), True),
                    pt.StructField('Unnamed: 0', pt.IntegerType(), True),
                    pt.StructField('id', pt.IntegerType(), True),
                    # cast below, Spark can't read 0/1 as bool
                    pt.StructField('bool_col', pt.ByteType(), True),
                    pt.StructField('tinyint_col', pt.ByteType(), True),
                    pt.StructField('smallint_col', pt.ShortType(), True),
                    pt.StructField('int_col', pt.IntegerType(), True),
                    pt.StructField('bigint_col', pt.LongType(), True),
                    pt.StructField('float_col', pt.FloatType(), True),
                    pt.StructField('double_col', pt.DoubleType(), True),
                    pt.StructField('date_string_col', pt.StringType(), True),
                    pt.StructField('string_col', pt.StringType(), True),
                    pt.StructField('timestamp_col', pt.TimestampType(), True),
                    pt.StructField('year', pt.IntegerType(), True),
                    pt.StructField('month', pt.IntegerType(), True),
                ]
            ),
            mode='FAILFAST',
            header=True,
        )
        .repartition(num_partitions)
        .sort('index')
    )

    df_functional_alltypes = df_functional_alltypes.withColumn(
        "bool_col", df_functional_alltypes["bool_col"].cast("boolean")
    )
    df_functional_alltypes.createOrReplaceTempView('functional_alltypes')

    df_batting = (
        s.read.csv(
            path=str(data_directory / 'batting.csv'),
            schema=pt.StructType(
                [
                    pt.StructField('playerID', pt.StringType(), True),
                    pt.StructField('yearID', pt.IntegerType(), True),
                    pt.StructField('stint', pt.IntegerType(), True),
                    pt.StructField('teamID', pt.StringType(), True),
                    pt.StructField('lgID', pt.StringType(), True),
                    pt.StructField('G', pt.IntegerType(), True),
                    pt.StructField('AB', pt.DoubleType(), True),
                    pt.StructField('R', pt.DoubleType(), True),
                    pt.StructField('H', pt.DoubleType(), True),
                    pt.StructField('X2B', pt.DoubleType(), True),
                    pt.StructField('X3B', pt.DoubleType(), True),
                    pt.StructField('HR', pt.DoubleType(), True),
                    pt.StructField('RBI', pt.DoubleType(), True),
                    pt.StructField('SB', pt.DoubleType(), True),
                    pt.StructField('CS', pt.DoubleType(), True),
                    pt.StructField('BB', pt.DoubleType(), True),
                    pt.StructField('SO', pt.DoubleType(), True),
                    pt.StructField('IBB', pt.DoubleType(), True),
                    pt.StructField('HBP', pt.DoubleType(), True),
                    pt.StructField('SH', pt.DoubleType(), True),
                    pt.StructField('SF', pt.DoubleType(), True),
                    pt.StructField('GIDP', pt.DoubleType(), True),
                ]
            ),
            header=True,
        )
        .repartition(num_partitions)
        .sort('playerID')
    )
    df_batting.createOrReplaceTempView('batting')

    df_awards_players = (
        s.read.csv(
            path=str(data_directory / 'awards_players.csv'),
            schema=pt.StructType(
                [
                    pt.StructField('playerID', pt.StringType(), True),
                    pt.StructField('awardID', pt.StringType(), True),
                    pt.StructField('yearID', pt.IntegerType(), True),
                    pt.StructField('lgID', pt.StringType(), True),
                    pt.StructField('tie', pt.StringType(), True),
                    pt.StructField('notes', pt.StringType(), True),
                ]
            ),
            header=True,
        )
        .repartition(num_partitions)
        .sort('playerID')
    )
    df_awards_players.createOrReplaceTempView('awards_players')

    df_simple = s.createDataFrame([(1, 'a')], ['foo', 'bar'])
    df_simple.createOrReplaceTempView('simple')

    df_struct = s.createDataFrame([((1, 2, 'a'),)], ['struct_col'])
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
        [
            (float(i), None if i % 2 else 3.0, 'ddeefffggh'[i])
            for i in range(10)
        ],
        ['a', 'b', 'key'],
    )
    df_udf_null.createOrReplaceTempView('udf_null')

    df_udf_random = s.createDataFrame(
        pd.DataFrame(
            {
                'a': np.arange(4, dtype=float).tolist()
                + np.random.rand(3).tolist(),
                'b': np.arange(4, dtype=float).tolist()
                + np.random.rand(3).tolist(),
                'key': list('ddeefff'),
            }
        )
    )
    df_udf_random.createOrReplaceTempView('udf_random')

    return _spark_testing_client


def get_pyspark_testing_client(data_directory):
    global _pyspark_testing_client
    if _pyspark_testing_client is None:
        _pyspark_testing_client = get_common_spark_testing_client(
            data_directory,
            lambda session: ibis.backends.pyspark.Backend().connect(session),
        )
    return _pyspark_testing_client


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

    df_nans = client._session.createDataFrame(
        [
            [np.NaN, 'Alfred', None],
            [27.0, 'Batman', 'motocycle'],
            [3.0, None, 'joker'],
        ],
        ['age', 'user', 'toy'],
    )
    df_nans.createTempView('nan_table')

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


def _random_identifier(suffix):
    return f'__ibis_test_{suffix}_{util.guid()}'


@pytest.fixture(scope='session', autouse=True)
def test_data_db(client):
    try:
        name = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')
        client.create_database(name)
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
    return client.table('functional_alltypes').relabel(
        {'Unnamed: 0': 'Unnamed:0'}
    )


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
