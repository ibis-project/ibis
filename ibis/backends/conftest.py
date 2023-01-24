from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any, TextIO

import _pytest
import pandas as pd
import pytest
import sqlalchemy as sa
from packaging.requirements import Requirement
from packaging.version import parse as vparse

import ibis
from ibis import util
from ibis.backends.base import _get_backend_names

TEST_TABLES = {
    "functional_alltypes": ibis.schema(
        {
            "index": "int64",
            "Unnamed: 0": "int64",
            "id": "int32",
            "bool_col": "boolean",
            "tinyint_col": "int8",
            "smallint_col": "int16",
            "int_col": "int32",
            "bigint_col": "int64",
            "float_col": "float32",
            "double_col": "float64",
            "date_string_col": "string",
            "string_col": "string",
            "timestamp_col": "timestamp",
            "year": "int32",
            "month": "int32",
        }
    ),
    "diamonds": ibis.schema(
        {
            "carat": "float64",
            "cut": "string",
            "color": "string",
            "clarity": "string",
            "depth": "float64",
            "table": "float64",
            "price": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
        }
    ),
    "batting": ibis.schema(
        {
            "playerID": "string",
            "yearID": "int64",
            "stint": "int64",
            "teamID": "string",
            "lgID": "string",
            "G": "int64",
            "AB": "int64",
            "R": "int64",
            "H": "int64",
            "X2B": "int64",
            "X3B": "int64",
            "HR": "int64",
            "RBI": "int64",
            "SB": "int64",
            "CS": "int64",
            "BB": "int64",
            "SO": "int64",
            "IBB": "int64",
            "HBP": "int64",
            "SH": "int64",
            "SF": "int64",
            "GIDP": "int64",
        }
    ),
    "awards_players": ibis.schema(
        {
            "playerID": "string",
            "awardID": "string",
            "yearID": "int64",
            "lgID": "string",
            "tie": "string",
            "notes": "string",
        }
    ),
}


@pytest.fixture(scope='session')
def script_directory() -> Path:
    """Return the test script directory.

    Returns
    -------
    Path
        Test script directory
    """
    return Path(__file__).absolute().parents[2] / "ci"


@pytest.fixture(scope='session')
def data_directory() -> Path:
    """Return the test data directory.

    Returns
    -------
    Path
        Test data directory
    """
    root = Path(__file__).absolute().parents[2]

    return root / "ci" / "ibis-testing-data"


def recreate_database(
    url: sa.engine.url.URL,
    database: str,
    **kwargs: Any,
) -> None:
    """Drop the `database` at `url`, if it exists.

    Create a new, blank database with the same name.

    Parameters
    ----------
    url : url.sa.engine.url.URL
        Connection url to the database
    database : str
        Name of the database to be dropped.
    """
    engine = sa.create_engine(url.set(database=""), **kwargs)

    if url.database is not None:
        with engine.begin() as con:
            con.exec_driver_sql(f"DROP DATABASE IF EXISTS {database}")
            con.exec_driver_sql(f"CREATE DATABASE {database}")


def init_database(
    url: sa.engine.url.URL,
    database: str,
    schema: TextIO | None = None,
    recreate: bool = True,
    isolation_level: str | None = "AUTOCOMMIT",
    **kwargs: Any,
) -> sa.engine.Engine:
    """Initialise `database` at `url` with `schema`.

    If `recreate`, drop the `database` at `url`, if it exists.

    Parameters
    ----------
    url : url.sa.engine.url.URL
        Connection url to the database
    database : str
        Name of the database to be dropped
    schema : TextIO
        File object containing schema to use
    recreate : bool
        If true, drop the database if it exists
    isolation_level : str
        Transaction isolation_level

    Returns
    -------
    sa.engine.Engine
        SQLAlchemy engine object
    """
    if isolation_level is not None:
        kwargs["isolation_level"] = isolation_level

    if recreate:
        recreate_database(url, database, **kwargs)

    try:
        url.database = database
    except AttributeError:
        url = url.set(database=database)

    engine = sa.create_engine(url, **kwargs)

    if schema:
        with engine.begin() as conn:
            for stmt in filter(None, map(str.strip, schema.read().split(';'))):
                conn.exec_driver_sql(stmt)

    return engine


def _random_identifier(suffix: str) -> str:
    return f"__ibis_test_{suffix}_{util.guid()}"


def _get_backend_conf(backend_str: str):
    """Convert a backend string to the test class for the backend."""
    conftest = importlib.import_module(f"ibis.backends.{backend_str}.tests.conftest")
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
    xdist_group_markers = []

    for item in items:
        parts = item.path.parts
        backend = _get_backend_from_parts(parts)
        if backend is not None and backend in all_backends:
            item.add_marker(getattr(pytest.mark, backend))
            item.add_marker(pytest.mark.backend)
        elif "backends" not in parts:
            # anything else is a "core" test and is run by default
            if not any(item.iter_markers(name="benchmark")):
                item.add_marker(pytest.mark.core)

        for name in ("duckdb", "sqlite"):
            # build a list of markers so we're don't invalidate the item's
            # marker iterator
            for _ in item.iter_markers(name=name):
                xdist_group_markers.append((item, pytest.mark.xdist_group(name=name)))

    for item, marker in xdist_group_markers:
        item.add_marker(marker)


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

    for marker in item.iter_markers(name="min_server_version"):
        kwargs = marker.kwargs
        if backend not in kwargs:
            continue

        funcargs = item.funcargs
        con = funcargs.get(
            "con",
            getattr(funcargs.get("backend"), "connection", None),
        )

        if con is None:
            continue

        min_server_version = kwargs.pop(backend)
        server_version = con.version
        condition = vparse(server_version) < vparse(min_server_version)
        item.add_marker(
            pytest.mark.xfail(
                condition,
                reason=(
                    f"unsupported functionality for server version {server_version}"
                ),
                **kwargs,
            )
        )

    for marker in item.iter_markers(name="min_version"):
        kwargs = marker.kwargs
        if backend not in kwargs:
            continue

        min_version = kwargs.pop(backend)
        reason = kwargs.pop("reason", None)
        version = getattr(importlib.import_module(backend), "__version__", None)
        if condition := version is None:  # pragma: no cover
            if reason is None:
                reason = f"{backend} backend module has no __version__ attribute"
        else:
            condition = vparse(version) < vparse(min_version)
            if reason is None:
                reason = f"test requires {backend}>={version}; got version {version}"
            else:
                reason = f"{backend}@{version} (<{min_version}): {reason}"
        item.add_marker(pytest.mark.xfail(condition, reason=reason, **kwargs))

    # Ibis hasn't exposed existing functionality
    # This xfails so that you know when it starts to pass
    for marker in item.iter_markers(name="notimpl"):
        if backend in marker.args[0]:
            reason = marker.kwargs.get("reason")
            item.add_marker(
                pytest.mark.xfail(
                    reason=reason or f'Feature not yet exposed in {backend}',
                    **{k: v for k, v in marker.kwargs.items() if k != "reason"},
                )
            )

    # Functionality is unavailable upstream (but could be)
    # This xfails so that you know when it starts to pass
    for marker in item.iter_markers(name="notyet"):
        if backend in marker.args[0]:
            reason = marker.kwargs.get("reason")
            item.add_marker(
                pytest.mark.xfail(
                    reason=reason or f'Feature not available upstream for {backend}',
                    **{k: v for k, v in marker.kwargs.items() if k != "reason"},
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
                    **{k: v for k, v in marker.kwargs.items() if k != "reason"},
                )
            )

    for marker in item.iter_markers(name="xfail_version"):
        kwargs = marker.kwargs
        if backend not in kwargs:
            continue

        provided_reason = kwargs.pop("reason", None)
        spec = kwargs.pop(backend)
        module = importlib.import_module(backend)
        version = getattr(module, "__version__", None)
        assert version is not None, f"{backend} module has no __version__ attribute"
        condition = Requirement(f"{backend}{spec}").specifier.contains(version)
        reason = f"{backend} backend test fails with {backend}{spec}"
        if provided_reason is not None:
            reason += f"; {provided_reason}"
        if condition:
            item.add_marker(pytest.mark.xfail(reason=reason, **kwargs))


@pytest.fixture(params=_get_backends_to_test(), scope='session')
def backend(request, data_directory, script_directory, tmp_path_factory, worker_id):
    """Return an instance of BackendTest, loaded with data."""

    cls = _get_backend_conf(request.param)
    return cls.load_data(data_directory, script_directory, tmp_path_factory, worker_id)


@pytest.fixture(scope="session")
def con(backend):
    """Instance of a backend client."""
    return backend.connection


def _setup_backend(
    request, data_directory, script_directory, tmp_path_factory, worker_id
):
    if (backend := request.param) == "duckdb" and platform.system() == "Windows":
        pytest.xfail(
            "windows prevents two connections to the same duckdb file "
            "even in the same process"
        )
        return None
    else:
        cls = _get_backend_conf(backend)
        return cls.load_data(
            data_directory, script_directory, tmp_path_factory, worker_id
        )


@pytest.fixture(
    params=_get_backends_to_test(discard=("dask", "pandas")),
    scope='session',
)
def ddl_backend(request, data_directory, script_directory, tmp_path_factory, worker_id):
    """Set up the backends that are SQL-based.

    (sqlite, postgres, mysql, duckdb, datafusion, clickhouse, pyspark,
    impala)
    """
    return _setup_backend(
        request, data_directory, script_directory, tmp_path_factory, worker_id
    )


@pytest.fixture(scope='session')
def ddl_con(ddl_backend):
    """Instance of Client, already connected to the db (if applies)."""
    return ddl_backend.connection


@pytest.fixture(
    params=_get_backends_to_test(
        keep=("sqlite", "postgres", "mysql", "duckdb", "snowflake")
    ),
    scope='session',
)
def alchemy_backend(
    request, data_directory, script_directory, tmp_path_factory, worker_id
):
    """Set up the SQLAlchemy-based backends.

    (sqlite, mysql, postgres, duckdb)
    """
    return _setup_backend(
        request, data_directory, script_directory, tmp_path_factory, worker_id
    )


@pytest.fixture(scope='session')
def alchemy_con(alchemy_backend):
    """Instance of Client, already connected to the db (if applies)."""
    return alchemy_backend.connection


@pytest.fixture(
    params=_get_backends_to_test(keep=("dask", "pandas", "pyspark")),
    scope='session',
)
def udf_backend(request, data_directory, script_directory, tmp_path_factory, worker_id):
    """Runs the UDF-supporting backends."""
    cls = _get_backend_conf(request.param)
    return cls.load_data(data_directory, script_directory, tmp_path_factory, worker_id)


@pytest.fixture(scope='session')
def udf_con(udf_backend):
    """Instance of Client, already connected to the db (if applies)."""
    return udf_backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes


@pytest.fixture(scope="session")
def json_t(backend):
    return backend.json_t


@pytest.fixture(scope='session')
def struct(backend):
    return backend.struct


@pytest.fixture(scope='session')
def sorted_alltypes(alltypes):
    return alltypes.order_by('id')


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
def struct_df(struct):
    return struct.execute()


@pytest.fixture(scope='session')
def udf_df(udf_alltypes):
    return udf_alltypes.execute()


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
    if geo is not None:
        return geo.execute(limit=None)
    return None


@pytest.fixture
def alchemy_temp_table(alchemy_con) -> str:
    """Return a temporary table name.

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
        with contextlib.suppress(NotImplementedError):
            alchemy_con.drop_table(name, force=True)


@pytest.fixture
def temp_table(con) -> str:
    """Return a temporary table name.

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
        with contextlib.suppress(NotImplementedError):
            con.drop_table(name, force=True)


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
        with contextlib.suppress(NotImplementedError):
            ddl_con.drop_view(name, force=True)


@pytest.fixture(scope='session')
def current_data_db(ddl_con) -> str:
    """Return current database name."""
    return ddl_con.current_database


@pytest.fixture
def alternate_current_database(ddl_con, ddl_backend, current_data_db: str) -> str:
    """Create a temporary database and yield its name. Drops the created
    database upon completion.

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
        pytest.skip(f"{ddl_backend.name()} doesn't have create_database method.")
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
