import abc
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.expr.types as ir


# TODO: Merge into BackendTest, #2564
class RoundingConvention:
    @staticmethod
    @abc.abstractmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        """Round a series to `decimals` number of decimal values."""


# TODO: Merge into BackendTest, #2564
class RoundAwayFromZero(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        if not decimals:
            return (
                -(np.sign(series)) * np.ceil(-(series.abs()) - 0.5)
            ).astype(np.int64)
        return series.round(decimals=decimals)


# TODO: Merge into BackendTest, #2564
class RoundHalfToEven(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        result = series.round(decimals=decimals)
        return result if decimals else result.astype(np.int64)


# TODO: Merge into BackendTest, #2564
class UnorderedComparator:
    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        left = left.sort_values().reset_index(drop=True)
        right = right.sort_values().reset_index(drop=True)
        return super().assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        columns = list(set(left.columns) & set(right.columns))
        left = left.sort_values(by=columns)
        right = right.sort_values(by=columns)
        return super().assert_frame_equal(left, right, *args, **kwargs)


class BackendTest(abc.ABC):
    check_dtype = True
    check_names = True
    supports_arrays = True
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    additional_skipped_operations = frozenset()
    supports_divide_by_zero = False
    returned_timestamp_unit = 'us'
    supported_to_timestamp_units = {'s', 'ms', 'us'}
    supports_floating_modulus = True

    def __init__(self, data_directory: Path) -> None:
        self.api  # skips if we can't access the backend
        self.connection = self.connect(data_directory)

    def __str__(self):
        return f'<BackendTest {self.name()}>'

    @classmethod
    def name(cls) -> str:
        backend_tests_path = inspect.getmodule(cls).__file__
        return Path(backend_tests_path).resolve().parent.parent.name

    @staticmethod
    @abc.abstractmethod
    def connect(data_directory: Path) -> ibis.client.Client:
        """Return a connection with data loaded from `data_directory`."""

    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        tm.assert_frame_equal(left, right, *args, **kwargs)

    @staticmethod
    def default_series_rename(
        series: pd.Series, name: str = 'tmp'
    ) -> pd.Series:
        return series.rename(name)

    @staticmethod
    def greatest(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        return f(*args)

    @staticmethod
    def least(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        return f(*args)

    @property
    def db(self) -> ibis.client.Database:
        return self.connection.database()

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        return self.db.functional_alltypes

    @property
    def batting(self) -> ir.TableExpr:
        return self.db.batting

    @property
    def awards_players(self) -> ir.TableExpr:
        return self.db.awards_players

    @property
    def geo(self) -> Optional[ir.TableExpr]:
        return None

    @property
    def api(self):
        return getattr(ibis.backends, self.name())

    def make_context(
        self, params: Optional[Mapping[ir.ValueExpr, Any]] = None
    ) -> comp.QueryContext:
        return self.api.dialect.make_context(params=params)


# TODO move to the spark/pyspark backends, #2565
_spark_testing_client = None
_pyspark_testing_client = None


# TODO move to the sparn/pyspark backends, #2565
def get_spark_testing_client(data_directory):
    global _spark_testing_client
    if _spark_testing_client is None:
        _spark_testing_client = get_common_spark_testing_client(
            data_directory,
            lambda session: ibis.backends.spark.connect(session),
        )
    return _spark_testing_client


# TODO move to the spark/pyspark backends, #2565
def get_pyspark_testing_client(data_directory):
    global _pyspark_testing_client
    if _pyspark_testing_client is None:
        _pyspark_testing_client = get_common_spark_testing_client(
            data_directory,
            lambda session: ibis.backends.pyspark.connect(session),
        )
    return _pyspark_testing_client


# TODO move to the spark/pyspark backends, #2565
def get_common_spark_testing_client(data_directory, connect):
    pytest.importorskip('pyspark')
    import pyspark.sql.types as pt
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    _spark_testing_client = connect(spark)
    s = _spark_testing_client._session

    df_functional_alltypes = s.read.csv(
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
    df_functional_alltypes = df_functional_alltypes.withColumn(
        "bool_col", df_functional_alltypes["bool_col"].cast("boolean")
    )
    df_functional_alltypes.createOrReplaceTempView('functional_alltypes')

    df_batting = s.read.csv(
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
    df_batting.createOrReplaceTempView('batting')

    df_awards_players = s.read.csv(
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
