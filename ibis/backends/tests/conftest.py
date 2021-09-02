import importlib
import os
from typing import List

import pandas as pd
import pkg_resources
import pytest

import ibis
import ibis.common.exceptions as com
import ibis.util as util

from .base import BackendTest


def _random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


def _get_all_backends() -> List[str]:
    """
    Return the list of known backend names.
    """
    return [
        entry_point.name
        for entry_point in pkg_resources.iter_entry_points(
            group='ibis.backends', name=None
        )
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


def _get_backends_to_test():
    """
    Get a list of `TestConf` classes of the backends to test.

    The list of backends can be specified by theuse rwith the `PYTEST_BACKENDS`
    environment variable, or otherwise all backends are being tested.
    """
    backends = os.environ.get('PYTEST_BACKENDS', '').split(' ')
    if backends == ['']:
        backends = _get_all_backends()

    return [
        pytest.param(
            _backend_name_to_class(backend),
            marks=getattr(pytest.mark, backend),
            id=backend,
        )
        for backend in sorted(backends)
    ]


def pytest_runtest_call(item):
    """Dynamically add various custom markers."""
    nodeid = item.nodeid
    backend = item.funcargs["backend"]
    assert isinstance(backend, BackendTest), "backend has type {!r}".format(
        type(backend).__name__
    )

    for marker in item.iter_markers(name="only_on_backends"):
        if backend.name() not in marker.args[0]:
            pytest.skip(
                f"only_on_backends: {backend} is not in {marker.args[0]} "
                f"{nodeid}"
            )

    for marker in item.iter_markers(name="skip_backends"):
        (backend_types,) = map(tuple, marker.args)
        if backend.name() in marker.args[0]:
            pytest.skip(f"skip_backends: {backend} {nodeid}")

    for marker in item.iter_markers(name="skip_missing_feature"):
        features = marker.args[0]
        missing_features = [
            feature for feature in features if not getattr(backend, feature)
        ]
        if missing_features:
            pytest.skip(
                f'Backend {backend} is missing features {missing_features} '
                f'needed to run {nodeid}'
            )

    for marker in item.iter_markers(name="xfail_backends"):
        if backend.name() in marker.args[0]:
            item.add_marker(
                pytest.mark.xfail(
                    reason=f'{backend} in xfail list: {marker.args[0]}',
                    **marker.kwargs,
                )
            )

    for marker in item.iter_markers(name="xpass_backends"):
        if backend.name() not in marker.args[0]:
            item.add_marker(
                pytest.mark.xfail(
                    reason=f'{backend} not in xpass list: {marker.args[0]}',
                    **marker.kwargs,
                )
            )

    for marker in item.iter_markers(name='min_spark_version'):
        min_version = marker.args[0]
        if backend.name() in ['spark', 'pyspark']:
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
        assert isinstance(
            backend, BackendTest
        ), "backend has type {!r}".format(type(backend).__name__)
        pytest.xfail(reason='{}: {}'.format(type(backend).__name__, e))


pytestmark = pytest.mark.backend


@pytest.fixture(params=_get_backends_to_test(), scope='session')
def backend(request, data_directory):
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    return backend.connection


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
        pytest.skip('Geo Spatial type not supported for {backend}.')
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
    # Currently geo is implemented just for OmniSciDB
    if geo is not None:
        return geo.execute(limit=None)
    return None


@pytest.fixture
def temp_table(con) -> str:
    """
    Return a temporary table name.

    Parameters
    ----------
    con : ibis.backends.base.Client

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
            f'{backend.name()} backend doesn\'t have current_database method.'
        )
    return con.current_database


@pytest.fixture
def alternate_current_database(con, backend, current_data_db: str) -> str:
    """Create a temporary database and yield its name.
    Drops the created database upon completion.

    Parameters
    ----------
    con : ibis.backends.base.Client
    current_data_db : str
    Yields
    -------
    str
    """
    name = _random_identifier('database')
    if not hasattr(con, 'create_database'):
        pytest.skip(
            f'{backend.name()} backend doesn\'t have create_database method.'
        )
    con.create_database(name)
    try:
        yield name
    finally:
        con.set_database(current_data_db)
        con.drop_database(name, force=True)


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
