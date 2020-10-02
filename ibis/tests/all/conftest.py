import operator
import os

import numpy as np
import pandas as pd
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.util as util
from ibis.tests.backends import Backend


def _random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


def subclasses(cls):
    """Get all child classes of `cls` not including `cls`, transitively."""
    assert isinstance(cls, type), "cls is not a class, type: {}".format(
        type(cls)
    )
    children = set(cls.__subclasses__())
    return children.union(*map(subclasses, children))


ALL_BACKENDS = sorted(subclasses(Backend), key=operator.attrgetter("__name__"))


def pytest_runtest_call(item):
    """Dynamically add various custom markers."""
    nodeid = item.nodeid
    for marker in list(item.iter_markers(name="only_on_backends")):
        (backend_types,) = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        if not isinstance(backend, backend_types):
            pytest.skip(
                f"only_on_backends: {backend} is not in {backend_types} "
                f"{nodeid}"
            )

    for marker in list(item.iter_markers(name="skip_backends")):
        (backend_types,) = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        if isinstance(backend, backend_types):
            pytest.skip(f"skip_backends: {backend} {nodeid}")

    for marker in list(item.iter_markers(name="skip_missing_feature")):
        backend = item.funcargs["backend"]
        (features,) = marker.args
        missing_features = [
            feature for feature in features if not getattr(backend, feature)
        ]
        if missing_features:
            pytest.mark.skip(
                'Backend {} is missing features {} needed to run {}'.format(
                    type(backend).__name__, ', '.join(missing_features), nodeid
                )
            )

    for marker in list(item.iter_markers(name="xfail_backends")):
        (backend_types,) = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        item.add_marker(
            pytest.mark.xfail(
                condition=isinstance(backend, backend_types),
                reason='Backend {} does not pass this test'.format(
                    type(backend).__name__
                ),
                **marker.kwargs,
            )
        )

    for marker in list(item.iter_markers(name="xpass_backends")):
        (backend_types,) = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        item.add_marker(
            pytest.mark.xfail(
                condition=not isinstance(backend, backend_types),
                reason='{} does not pass this test'.format(
                    type(backend).__name__
                ),
                **marker.kwargs,
            )
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Dynamically add an xfail marker for specific backends."""
    outcome = yield
    try:
        outcome.get_result()
    except (
        com.OperationNotDefinedError,
        com.UnsupportedOperationError,
        com.UnsupportedBackendType,
        NotImplementedError,
    ) as e:
        markers = list(pyfuncitem.iter_markers(name="xfail_unsupported"))
        assert (
            len(markers) == 1
        ), "More than one xfail_unsupported marker found on test {}".format(
            pyfuncitem
        )
        (marker,) = markers
        backend = pyfuncitem.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        pytest.xfail(reason='{}: {}'.format(type(backend).__name__, e))


pytestmark = pytest.mark.backend

pytest_backends = os.environ.get('PYTEST_BACKENDS', '').split(' ')
params_backend = [
    pytest.param(backend, marks=getattr(pytest.mark, backend.__name__.lower()))
    for backend in ALL_BACKENDS
    if backend.__name__.lower() in pytest_backends or not pytest_backends
]
if len(pytest_backends) != len(params_backend):
    unknown_backends = set(pytest_backends) - set(
        b.__name__.lower() for b in ALL_BACKENDS
    )
    raise ValueError(
        'PYTEST_BACKENDS environment variable contain unknown '
        f'backends {unknown_backends}'
    )


@pytest.fixture(params=params_backend, scope='session')
def backend(request, data_directory):
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    return backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes


@pytest.fixture(scope='session')
def sorted_alltypes(alltypes):
    return alltypes.sort_by('id')


@pytest.fixture(scope='session')
def batting(backend):
    return backend.batting


@pytest.fixture(scope='session')
def awards_players(backend):
    return backend.awards_players


@pytest.fixture(scope='session')
def geo(backend):
    if backend.geo is None:
        pytest.skip(
            'Geo Spatial type not supported for {} backend.'.format(
                backend.name
            )
        )
    return backend.geo


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def sorted_df(df):
    return df.sort_values('id').reset_index(drop=True)


@pytest.fixture(scope='session')
def batting_df(batting):
    return batting.execute(limit=None)


@pytest.fixture(scope='session')
def awards_players_df(awards_players):
    return awards_players.execute(limit=None)


@pytest.fixture(scope='session')
def geo_df(geo):
    # Currently geo is implemented just for OmniSciDB
    if geo is not None:
        return geo.execute(limit=None)
    return None


_spark_testing_client = None
_pyspark_testing_client = None


def get_spark_testing_client(data_directory):
    global _spark_testing_client
    if _spark_testing_client is None:
        _spark_testing_client = get_common_spark_testing_client(
            data_directory, lambda session: ibis.spark.connect(session)
        )
    return _spark_testing_client


def get_pyspark_testing_client(data_directory):
    global _pyspark_testing_client
    if _pyspark_testing_client is None:
        _pyspark_testing_client = get_common_spark_testing_client(
            data_directory, lambda session: ibis.pyspark.connect(session)
        )
    return _pyspark_testing_client


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


@pytest.fixture
def temp_table(con: ibis.client.Client) -> str:
    """
    Return a temporary table name.

    Parameters
    ----------
    con : ibis.client.Client

    Yields
    ------
    name : string
        Random table name for a temporary usage.
    """
    name = _random_identifier('table')
    try:
        yield name
    finally:
        if hasattr(con, 'drop_table'):
            con.drop_table(name, force=True)


@pytest.fixture
def temp_view(con) -> str:
    """Return a temporary view name.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Yields
    ------
    name : string
        Random view name for a temporary usage.
    """
    name = _random_identifier('view')
    try:
        yield name
    finally:
        if hasattr(con, 'drop_view'):
            con.drop_view(name, force=True)


@pytest.fixture(scope='session')
def current_data_db(con, backend) -> str:
    """Return current database name."""
    if not hasattr(con, 'current_database'):
        pytest.skip(
            f'{backend.name} backend doesn\'t have current_database method.'
        )
    return con.current_database


@pytest.fixture
def alternate_current_database(con, backend, current_data_db: str) -> str:
    """Create a temporary database and yield its name.
    Drops the created database upon completion.

    Parameters
    ----------
    con : ibis.client.Client
    current_data_db : str
    Yields
    -------
    str
    """
    name = _random_identifier('database')
    if not hasattr(con, 'create_database'):
        pytest.skip(
            f'{backend.name} backend doesn\'t have create_database method.'
        )
    con.create_database(name)
    try:
        yield name
    finally:
        con.set_database(current_data_db)
        con.drop_database(name, force=True)
