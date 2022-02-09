import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.util as util

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

from .tests.base import BackendTest


@pytest.fixture(scope='session')
def data_directory() -> Path:
    """
    Fixture that returns the test data directory.

    Returns
    -------
    Path
        Test data directory
    """
    root = Path(__file__).absolute().parent.parent.parent

    default = root / 'ci' / 'ibis-testing-data'
    datadir = os.environ.get('IBIS_TEST_DATA_DIRECTORY', default)
    datadir = Path(datadir)

    return datadir


@pytest.fixture
def file_backends_data():
    # basic time/ticker frame
    rng = pd.date_range('20170101', periods=10, freq='D')
    tickers = ['GOOGL', 'FB', 'APPL', 'NFLX', 'AMZN']
    df = pd.DataFrame(
        {
            'time': np.repeat(rng, len(tickers)),
            'ticker': np.tile(tickers, len(rng)),
        }
    )
    opens = df.assign(open=np.random.randn(len(df)))
    closes = df.assign(close=np.random.randn(len(df)))
    return {'open': opens, 'close': closes}


def _random_identifier(suffix):
    return f'__ibis_test_{suffix}_{util.guid()}'


def _get_all_backends() -> List[str]:
    """
    Return the list of known backend names.
    """
    return [
        entry_point.name
        for entry_point in importlib_metadata.entry_points()["ibis.backends"]
    ]


def _backend_name_to_class(backend_str: str):
    """
    Convert a backend string to the test configuration class for the backend.
    """
    try:
        backend_package = getattr(ibis, backend_str).__module__
    except AttributeError:
        raise ValueError(
            f'Unknown backend {backend_str}. '
            f'Known backends: {_get_all_backends()}'
        )

    conftest = importlib.import_module(f'{backend_package}.tests.conftest')
    return conftest.TestConf


@lru_cache(maxsize=None)
def _get_backends_to_test(keep=None, discard=None):
    """
    Get a list of `TestConf` classes of the backends to test.

    The list of backends can be specified by the user with the
    `PYTEST_BACKENDS` environment variable.

    - If the variable is undefined or empty, then no backends are returned
    - Otherwise the variable must contain a space-separated list of backends to
      test

    """
    backends_raw = os.environ.get('PYTEST_BACKENDS')

    if not backends_raw:
        return []

    if backends_raw == "all":
        backends = set(_get_all_backends())
        # spark is just an alias to pyspark so don't re-run all spark tests
        backends.discard("spark")
    else:
        backends = set(backends_raw.split())

    if discard is not None:
        backends = backends.difference(discard.split())

    if keep is not None:
        backends = backends.intersection(keep.split())

    backends = sorted(list(backends))

    return [
        pytest.param(
            _backend_name_to_class(backend),
            marks=[getattr(pytest.mark, backend), pytest.mark.backend],
            id=backend,
        )
        for backend in sorted(backends)
    ]


def pytest_runtest_call(item):
    """Dynamically add various custom markers."""
    nodeid = item.nodeid
    backend = [
        backend.name()
        for key, backend in item.funcargs.items()
        if key.endswith("backend")
    ]
    if len(backend) > 1:
        breakpoint()
        raise ValueError(
            f"test {item.originalname} was supplied with multiple backend "
            f"objects simultaneously. This is likely due to a leaky fixture."
            f"Backends passed: {(back.name() for back in backend)}"
        )
    if not backend:
        # Check item path to see if test is in backend-specific folder
        backend = set(_get_all_backends()).intersection(item.path.parts)

    if not backend:
        return

    backend = next(iter(backend))

    for marker in item.iter_markers(name="only_on_backends"):
        if backend not in marker.args[0]:
            pytest.skip(
                f"only_on_backends: {backend} is not in {marker.args[0]} "
                f"{nodeid}"
            )

    for marker in item.iter_markers(name="skip_backends"):
        if backend in marker.args[0]:
            pytest.skip(f"skip_backends: {backend} {nodeid}")

    for marker in item.iter_markers(name="xfail_backends"):
        for entry in marker.args[0]:
            if isinstance(entry, tuple):
                name, reason = entry
            else:
                name = entry
                reason = marker.kwargs.get("reason")

            if backend == name:
                item.add_marker(
                    pytest.mark.xfail(
                        reason=reason or f'{backend} in xfail list: {name}',
                        **{
                            k: v
                            for k, v in marker.kwargs.items()
                            if k != "reason"
                        },
                    )
                )

    for marker in item.iter_markers(name="xpass_backends"):
        if backend not in marker.args[0]:
            item.add_marker(
                pytest.mark.xfail(
                    reason=f'{backend} not in xpass list: {marker.args[0]}',
                    **marker.kwargs,
                )
            )

    for marker in item.iter_markers(name='min_spark_version'):
        min_version = marker.args[0]
        if backend in ['spark', 'pyspark']:
            from distutils.version import LooseVersion

            import pyspark

            if LooseVersion(pyspark.__version__) < LooseVersion(min_version):
                item.add_marker(
                    pytest.mark.xfail(
                        reason=f'Require minimal spark version {min_version}, '
                        f'but is {pyspark.__version__}',
                        **marker.kwargs,
                    )
                )

    # Ibis hasn't exposed existing functionality
    # This xfails so that you know when it starts to pass
    for marker in item.iter_markers(name="notimpl"):
        if backend in marker.args[0]:
            reason = marker.kwargs.get("reason")
            item.add_marker(
                pytest.mark.xfail(
                    reason=reason or f'Feature not yet exposed in {backend}',
                    **{
                        k: v for k, v in marker.kwargs.items() if k != "reason"
                    },
                )
            )

    # Functionality is unavailable upstream (but could be)
    # This xfails so that you know when it starts to pass
    for marker in item.iter_markers(name="notyet"):
        if backend in marker.args[0]:
            reason = marker.kwargs.get("reason")
            item.add_marker(
                pytest.mark.xfail(
                    reason=reason
                    or f'Feature not available upstream for {backend}',
                    **{
                        k: v for k, v in marker.kwargs.items() if k != "reason"
                    },
                )
            )

    for marker in item.iter_markers(name="never"):
        if backend in marker.args[0]:
            if "reason" not in marker.kwargs.keys():
                raise ValueError("never requires a reason")
            item.add_marker(
                pytest.mark.xfail(
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
        backend = pyfuncitem.funcargs.get("backend", None)
        if backend is None:
            return
        backend_type = type(backend).__name__

        if len(markers) == 0:
            # nothing has marked the failure as an expected one
            raise e
        elif len(markers) == 1:
            if not isinstance(backend, BackendTest):
                pytest.fail(f"Backend has type {backend_type!r}")
            pytest.xfail(reason=f"{backend_type!r}: {e}")
        else:
            pytest.fail(
                f"More than one xfail_unsupported marker found on test "
                f"{pyfuncitem}"
            )


pytestmark = pytest.mark.backend


@pytest.fixture(params=_get_backends_to_test(), scope='session')
def backend(request, data_directory):
    """
    Instance of BackendTest.

    (dask pandas clickhouse sqlite postgres pyspark impala hdf5 csv parquet
    datafusion mysql)
    """
    # See #3021
    # TODO Remove this to backend_test, since now that a `Backend` class exists
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    """
    Instance of Client, already connected to the db (if applies).
    """
    # See #3021
    # TODO Rename this to `backend` when the existing `backend` is renamed to
    # `backend_test`, and when `connect` returns `Backend` and not `Client`
    return backend.connection


@pytest.fixture(
    params=_get_backends_to_test(discard="dask csv parquet hdf5 pandas"),
    scope='session',
)
def ddl_backend(request, data_directory):
    """
    Runs the SQL-ish backends
    (sqlite, postgres, mysql, datafusion, clickhouse, pyspark, impala)
    """
    return request.param(data_directory)


@pytest.fixture(scope='session')
def ddl_con(ddl_backend):
    """
    Instance of Client, already connected to the db (if applies).
    """
    return ddl_backend.connection


@pytest.fixture(
    params=_get_backends_to_test(keep="sqlite postgres mysql"),
    scope='session',
)
def alchemy_backend(request, data_directory):
    """
    Runs the SQLAlchemy-based backends
    (sqlite, mysql, postgres)
    """
    return request.param(data_directory)


@pytest.fixture(scope='session')
def alchemy_con(alchemy_backend):
    """
    Instance of Client, already connected to the db (if applies).
    """
    return alchemy_backend.connection


@pytest.fixture(
    params=_get_backends_to_test(discard="csv parquet hdf5"),
    scope='session',
)
def rw_backend(request, data_directory):
    """
    Runs the non-file-based backends
    (dask pandas clickhouse sqlite postgres pyspark impala datafusion mysql)
    """
    return request.param(data_directory)


@pytest.fixture(scope='session')
def rw_con(rw_backend):
    """
    Instance of Client, already connected to the db (if applies).
    """
    return rw_backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes


@pytest.fixture(scope='session')
def sorted_alltypes(backend, alltypes):
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
        pytest.skip(f'Geo Spatial type not supported for {backend}.')
    return backend.geo


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def sorted_df(backend, df):
    return df.sort_values('id').reset_index(drop=True)


@pytest.fixture(scope='session')
def batting_df(batting):
    return batting.execute(limit=None)


@pytest.fixture(scope='session')
def awards_players_df(awards_players):
    return awards_players.execute(limit=None)


@pytest.fixture(scope='session')
def geo_df(geo):
    if geo is not None:
        return geo.execute(limit=None)
    return None


@pytest.fixture
def alchemy_temp_table(alchemy_con) -> str:
    """
    Return a temporary table name.

    Parameters
    ----------
    alchemy_con : ibis.backends.base.Client

    Yields
    ------
    name : string
        Random table name for a temporary usage.
    """
    name = _random_identifier('table')
    try:
        yield name
    finally:
        try:
            alchemy_con.drop_table(name, force=True)
        except NotImplementedError:
            pass


@pytest.fixture
def temp_table(rw_con) -> str:
    """
    Return a temporary table name.

    Parameters
    ----------
    rw_con : ibis.backends.base.Client

    Yields
    ------
    name : string
        Random table name for a temporary usage.
    """
    name = _random_identifier('table')
    try:
        yield name
    finally:
        try:
            rw_con.drop_table(name, force=True)
        except NotImplementedError:
            pass


@pytest.fixture
def temp_view(ddl_con) -> str:
    """Return a temporary view name.

    Parameters
    ----------
    ddl_con : backend connection

    Yields
    ------
    name : string
        Random view name for a temporary usage.
    """
    name = _random_identifier('view')
    try:
        yield name
    finally:
        try:
            ddl_con.drop_view(name, force=True)
        except NotImplementedError:
            pass


@pytest.fixture(scope='session')
def current_data_db(ddl_con, ddl_backend) -> str:
    """Return current database name."""
    return ddl_con.current_database


@pytest.fixture
def alternate_current_database(
    ddl_con, ddl_backend, current_data_db: str
) -> str:
    """Create a temporary database and yield its name.
    Drops the created database upon completion.

    Parameters
    ----------
    ddl_con : ibis.backends.base.Client
    current_data_db : str
    Yields
    -------
    str
    """
    name = _random_identifier('database')
    try:
        ddl_con.create_database(name)
    except NotImplementedError:
        pytest.skip(
            f"{ddl_backend.name()} doesn't have create_database method."
        )
    try:
        yield name
    finally:
        ddl_con.set_database(current_data_db)
        ddl_con.drop_database(name, force=True)


@pytest.fixture
def test_employee_schema() -> ibis.schema:
    sch = ibis.schema(
        [
            ('first_name', 'string'),
            ('last_name', 'string'),
            ('department_name', 'string'),
            ('salary', 'float64'),
        ]
    )

    return sch


@pytest.fixture
def test_employee_data_1():
    df = pd.DataFrame(
        {
            'first_name': ['A', 'B', 'C'],
            'last_name': ['D', 'E', 'F'],
            'department_name': ['AA', 'BB', 'CC'],
            'salary': [100.0, 200.0, 300.0],
        }
    )

    return df


@pytest.fixture
def test_employee_data_2():
    df2 = pd.DataFrame(
        {
            'first_name': ['X', 'Y', 'Z'],
            'last_name': ['A', 'B', 'C'],
            'department_name': ['XX', 'YY', 'ZZ'],
            'salary': [400.0, 500.0, 600.0],
        }
    )

    return df2
