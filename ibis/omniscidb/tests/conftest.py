"""OmniSciDB test configuration module."""
import os
import typing

import pandas
import pytest

import ibis
from ibis.tests.util import TempHelper

OMNISCIDB_HOST = os.environ.get('IBIS_TEST_OMNISCIDB_HOST', 'localhost')
OMNISCIDB_PORT = int(os.environ.get('IBIS_TEST_OMNISCIDB_PORT', 6274))
OMNISCIDB_USER = os.environ.get('IBIS_TEST_OMNISCIDB_USER', 'admin')
OMNISCIDB_PASS = os.environ.get(
    'IBIS_TEST_OMNISCIDB_PASSWORD', 'HyperInteractive'
)
OMNISCIDB_PROTOCOL = os.environ.get('IBIS_TEST_OMNISCIDB_PROTOCOL', 'binary')
OMNISCIDB_DB = os.environ.get('IBIS_TEST_DATA_DB', 'ibis_testing')


@pytest.fixture(scope='module')
def con():
    """Define a connection fixture.

    Returns
    -------
    ibis.omniscidb.OmniSciDBClient
    """
    return ibis.omniscidb.connect(
        protocol=OMNISCIDB_PROTOCOL,
        host=OMNISCIDB_HOST,
        port=OMNISCIDB_PORT,
        user=OMNISCIDB_USER,
        password=OMNISCIDB_PASS,
        database=OMNISCIDB_DB,
    )


@pytest.fixture
def schema() -> ibis.schema:
    """
    Define fixture for test schema

    Returns
    -------
    ibis.expr.Schema
    """
    return ibis.schema(
        [('a', 'polygon'), ('b', 'point'), ('c', 'int8'), ('d', 'double')]
    )


@pytest.fixture
def table(con, schema):
    """
    Define fixture for test table.

    Yields
    -------
    ibis.expr.types.TableExpr
    """
    with TempHelper(
        con, kind='table', create=True, create_kwargs={'schema': schema}
    ) as name:
        yield con.table(name)


@pytest.fixture(scope='module')
def session_con():
    """Define a session connection fixture."""
    # TODO: fix return issue
    return ibis.omniscidb.connect(
        protocol=OMNISCIDB_PROTOCOL,
        host=OMNISCIDB_HOST,
        port=OMNISCIDB_PORT,
        user=OMNISCIDB_USER,
        password=OMNISCIDB_PASS,
        database=OMNISCIDB_DB,
    )
    return session_con


@pytest.fixture(scope='module')
def alltypes(con) -> ibis.expr.types.TableExpr:
    """Define a functional_alltypes table fixture.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Returns
    -------
    ibis.expr.types.TableExpr
    """
    return con.table('functional_alltypes')


@pytest.fixture(scope='module')
def awards_players(con) -> ibis.expr.types.TableExpr:
    """Define a awards_players table fixture.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Returns
    -------
    ibis.expr.types.TableExpr
    """
    return con.table('awards_players')


@pytest.fixture(scope='module')
def batting(con) -> ibis.expr.types.TableExpr:
    """Define a awards_players table fixture.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Returns
    -------
    ibis.expr.types.TableExpr
    """
    return con.table('batting')


@pytest.fixture(scope='module')
def df_alltypes(alltypes: ibis.expr.types.TableExpr) -> pandas.DataFrame:
    """Return all the data for functional_alltypes table.

    Parameters
    ----------
    alltypes : ibis.expr.types.TableExpr
        [description]

    Returns
    -------
    pandas.DataFrame
    """
    return alltypes.execute()


@pytest.fixture
def translate() -> typing.Callable:
    """Create a translator function.

    Returns
    -------
    function
    """
    from ibis.omniscidb.compiler import OmniSciDBDialect

    dialect = OmniSciDBDialect()
    context = dialect.make_context()
    return lambda expr: dialect.translator(expr, context).get_result()


@pytest.fixture
def table_name(con) -> str:
    """Return a temporary table name.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Yields
    ------
    name : string
        Random table name for a temporary usage.
    """
    with TempHelper(
        con, kind='table', create=False, create_kwargs={'schema': schema}
    ) as result:
        yield result


@pytest.fixture(scope='session')
def test_data_db() -> str:
    """Return the database name."""
    return OMNISCIDB_DB


@pytest.fixture
def database(con, test_data_db: str) -> str:
    """Create a temporary database.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient
    test_data_db : str

    Yields
    -------
    str
    """
    with TempHelper(con, kind='database', create=True) as result:
        yield result


@pytest.fixture
def view(con) -> str:
    """Return a temporary view name.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Yields
    ------
    name : string
        Random view name for a temporary usage.
    """
    with TempHelper(con, kind='view') as result:
        yield result


def new_view(con, table) -> str:
    """Create a temporary view.
    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient
    Yields
    -------
    str
    """
    with TempHelper(
        con, kind='view', create=True, create_kwargs={'expr': table}
    ) as result:
        yield result


@pytest.fixture
def user(con) -> list:
    """Return a list with temporary user name and password

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Yields
    -------
    tupple
    """
    with TempHelper(con, kind="user") as name:
        yield (name, "super")
