import os
from pathlib import Path

import pytest

import ibis

collect_ignore = ['setup.py']


@pytest.fixture(scope='session')
def data_directory():
    root = Path(__file__).absolute().parent

    default = root / 'ci' / 'ibis-testing-data'
    datadir = os.environ.get('IBIS_TEST_DATA_DIRECTORY', default)
    datadir = Path(datadir)

    if not datadir.exists():
        pytest.skip('test data directory not found')

    return datadir


@pytest.fixture(scope='session')
def spark_client_testing(data_directory):
    pytest.importorskip('pyspark')

    import pyspark.sql.types as pt

    client = ibis.spark.connect()

    df_functional_alltypes = client._session.read.csv(
        path=str(data_directory / 'functional_alltypes.csv'),
        schema=pt.StructType([
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
        ]),
        mode='FAILFAST',
        header=True,
    )
    df_functional_alltypes = df_functional_alltypes.withColumn(
        "bool_col", df_functional_alltypes["bool_col"].cast("boolean"))
    df_functional_alltypes.createOrReplaceTempView('functional_alltypes')

    df_batting = client._session.read.csv(
        path=str(data_directory / 'batting.csv'),
        schema=pt.StructType([
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
        ]),
        header=True,
    )
    df_batting.createOrReplaceTempView('batting')

    df_awards_players = client._session.read.csv(
        path=str(data_directory / 'awards_players.csv'),
        schema=pt.StructType([
            pt.StructField('playerID', pt.StringType(), True),
            pt.StructField('awardID', pt.StringType(), True),
            pt.StructField('yearID', pt.IntegerType(), True),
            pt.StructField('lgID', pt.StringType(), True),
            pt.StructField('tie', pt.StringType(), True),
            pt.StructField('notes', pt.StringType(), True),
        ]),
        header=True,
    )
    df_awards_players.createOrReplaceTempView('awards_players')

    df_simple = client._session.createDataFrame([(1, 'a')], ['foo', 'bar'])
    df_simple.createOrReplaceTempView('simple')

    df_struct = client._session.createDataFrame(
        [((1, 2, 'a'),)],
        ['struct_col']
    )
    df_struct.createOrReplaceTempView('struct')

    df_nested_types = client._session.createDataFrame(
        [
            (
                [1, 2],
                [[3, 4], [5, 6]],
                {'a' : [[2, 4], [3, 5]]},
            )
        ],
        [
            'list_of_ints',
            'list_of_list_of_ints',
            'map_string_list_of_list_of_ints'
        ]
    )
    df_nested_types.createOrReplaceTempView('nested_types')

    df_complicated = client._session.createDataFrame(
        [({(1, 3) : [[2, 4], [3, 5]]},)],
        ['map_tuple_list_of_list_of_ints']
    )
    df_complicated.createOrReplaceTempView('complicated')

    return client
