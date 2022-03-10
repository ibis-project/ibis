from __future__ import annotations

import importlib
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

import _pytest
import pandas as pd
import pytest

import ibis
import ibis.util as util

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


@pytest.fixture(scope='session')
def data_directory() -> Path:
    """Return the test data directory.

    Returns
    -------
    Path
        Test data directory
    """
    root = Path(__file__).absolute().parent.parent.parent

    return Path(
        os.environ.get(
            "IBIS_TEST_DATA_DIRECTORY",
            root / "ci" / "ibis-testing-data",
        )
    )


def _random_identifier(suffix: str) -> str:
    return f"__ibis_test_{suffix}_{util.guid()}"


@lru_cache(maxsize=None)
def _get_backend_names() -> frozenset[str]:
    """Return the set of known backend names.

    Notes
    -----
    This function returns a frozenset to prevent cache pollution.

    If a `set` is used, then any in-place modifications to the set
    are visible to every caller of this function.
    """
    return frozenset(
        entry_point.name
        for entry_point in importlib_metadata.entry_points()["ibis.backends"]
    )


def _get_backend_conf(backend_str: str):
    """Convert a backend string to the test class for the backend."""
    conftest = importlib.import_module(
        f"ibis.backends.{backend_str}.tests.conftest"
    )
    return conftest.TestConf


def _get_backend_from_parts(parts: tuple[str, ...]) -> str | None:
    """Return the backend part of a test file's path parts.

    Examples
    --------
    >>> _get_backend_from_parts(("/", "ibis", "backends", "sqlite", "tests"))
    "sqlite"
    """
    try:
        index = parts.index("backends")
    except ValueError:
        return None
    else:
        return parts[index + 1]


def pytest_ignore_collect(path, config):
    # get the backend path part
    #
    # path is a py.path.local object hence the conversion to Path first
    backend = _get_backend_from_parts(Path(path).parts)
    if backend is None or backend not in _get_backend_names():
        return False

    # we evaluate the marker early so that we don't trigger
    # an import of conftest files for the backend, which will
    # import the backend and thus require dependencies that may not
    # exist
    #
    # alternatives include littering library code with pytest.importorskips
    # and moving all imports close to their use site
    #
    # the latter isn't tenable for backends that use multiple dispatch
    # since the rules are executed at import time
    mark_expr = config.getoption("-m")
    # we can't let the empty string pass through, since `'' in s` is `True` for
    # any `s`; if no marker was passed don't ignore the collection of `path`
    if not mark_expr:
        return False
    expr = _pytest.mark.expression.Expression.compile(mark_expr)
    # we check the "backend" marker as well since if that's passed
    # any file matching a backed should be skipped
    keep = expr.evaluate(lambda s: s in (backend, "backend"))
    return not keep


def pytest_collection_modifyitems(session, config, items):
    # add the backend marker to any tests are inside "ibis/backends"
    all_backends = _get_backend_names()
    for item in items:
        parts = item.path.parts
        backend = _get_backend_from_parts(parts)
        if backend is not None and backend in all_backends:
            item.add_marker(getattr(pytest.mark, backend))
            item.add_marker(pytest.mark.backend)
        elif "backends" not in parts:
            # anything else is a "core" test and is run by default
            item.add_marker(pytest.mark.core)

        if "sqlite" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group(name="sqlite"))
        if "duckdb" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group(name="duckdb"))


@lru_cache(maxsize=None)
def _get_backends_to_test(
    keep: tuple[str, ...] = (),
    discard: tuple[str, ...] = (),
) -> list[Any]:
    """Get a list of `TestConf` classes of the backends to test."""
    backends = _get_backend_names()

    if discard:
        backends = backends.difference(discard)

    if keep:
        backends = backends.intersection(keep)

    # spark is an alias for pyspark
    backends = backends.difference(("spark",))

    return [
        pytest.param(
            backend,
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
        raise ValueError(
            f"test {item.originalname} was supplied with multiple backend "
            f"objects simultaneously. This is likely due to a leaky fixture."
            f"Backends passed: {(back.name() for back in backend)}"
        )
    if not backend:
        # Check item path to see if test is in backend-specific folder
        backend = set(_get_backend_names()).intersection(item.path.parts)

    if not backend:
        return

    backend = next(iter(backend))

    for marker in item.iter_markers(name="skip_backends"):
        if backend in marker.args[0]:
            pytest.skip(f"skip_backends: {backend} {nodeid}")

    for marker in item.iter_markers(name='min_spark_version'):
        min_version = marker.args[0]
        if backend == 'pyspark':
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

    # Something has been exposed as broken by a new test and it shouldn't be
    # imperative for a contributor to fix it just because they happened to
    # bring it to attention  -- USE SPARINGLY
    for marker in item.iter_markers(name="broken"):
        if backend in marker.args[0]:
            reason = marker.kwargs.get("reason")
            item.add_marker(
                pytest.mark.xfail(
                    reason=reason or f"Feature is failing on {backend}",
                    **{
                        k: v for k, v in marker.kwargs.items() if k != "reason"
                    },
                )
            )


@pytest.fixture(params=_get_backends_to_test(), scope='session')
def backend(request, data_directory):
    """Return an instance of BackendTest."""
    cls = _get_backend_conf(request.param)
    return cls(data_directory)


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
    params=_get_backends_to_test(discard=("dask", "pandas")),
    scope='session',
)
def ddl_backend(request, data_directory):
    """
    Runs the SQL-ish backends
    (sqlite, postgres, mysql, datafusion, clickhouse, pyspark, impala)
    """
    cls = _get_backend_conf(request.param)
    return cls(data_directory)


@pytest.fixture(scope='session')
def ddl_con(ddl_backend):
    """
    Instance of Client, already connected to the db (if applies).
    """
    return ddl_backend.connection


@pytest.fixture(
    params=_get_backends_to_test(
        keep=("sqlite", "postgres", "mysql", "duckdb")
    ),
    scope='session',
)
def alchemy_backend(request, data_directory):
    """
    Runs the SQLAlchemy-based backends
    (sqlite, mysql, postgres)
    """
    if request.param == "duckdb" and platform.system() == "Windows":
        pytest.xfail(
            "windows prevents two connections to the same duckdb file "
            "even in the same process"
        )
    else:
        cls = _get_backend_conf(request.param)
        return cls(data_directory)


@pytest.fixture(scope='session')
def alchemy_con(alchemy_backend):
    """
    Instance of Client, already connected to the db (if applies).
    """
    return alchemy_backend.connection


@pytest.fixture(
    params=_get_backends_to_test(keep=("dask", "pandas", "pyspark")),
    scope='session',
)
def udf_backend(request, data_directory):
    """
    Runs the UDF-supporting backends
    """
    cls = _get_backend_conf(request.param)
    return cls(data_directory)


@pytest.fixture(scope='session')
def udf_con(udf_backend):
    """
    Instance of Client, already connected to the db (if applies).
    """
    return udf_backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes


@pytest.fixture(scope='session')
def sorted_alltypes(backend, alltypes):
    return alltypes.sort_by('id')


@pytest.fixture(scope='session')
def udf_alltypes(udf_backend):
    return udf_backend.functional_alltypes


@pytest.fixture(scope='session')
def batting(backend):
    return backend.batting


@pytest.fixture(scope='session')
def awards_players(backend):
    return backend.awards_players


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def udf_df(udf_alltypes):
    return udf_alltypes.execute()


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
        try:
            con.drop_table(name, force=True)
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
