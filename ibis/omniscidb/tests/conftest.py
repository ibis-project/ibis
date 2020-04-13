"""OmniSciDB test configuration module."""
import os
import typing

import pandas
import pytest

import ibis
import ibis.util as util

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


class temp_helper:
    def __init__(
        self,
        con,
        kind,
        create=False,
        create_kwargs=dict(),
        extra_return=[],
        method_name=None,
    ):
        self.name = _random_identifier(kind)
        self.con = con
        self.kind = kind
        self.create = create
        self.create_kwargs = create_kwargs
        self.extra_return = extra_return
        self.method_name = kind if method_name is None else method_name

    def __enter__(self):
        # some of the temp entities may not support 'force' parameter
        # at 'drop' method, that's why we use 'try-except' for that
        try:
            getattr(self.con, 'drop_' + self.method_name)(self.name)
        except Exception:
            pass

        if self.create:
            getattr(self.con, 'create_' + self.method_name)(
                self.name, **self.create_kwargs
            )
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            getattr(self.con, 'drop_' + self.method_name)(self.name)
        except Exception:
            pass


@pytest.fixture(scope='function')
def test_schema() -> ibis.schema:
    """
    Define fixture for test schema

    Returns
    -------
    ibis.expr.Schema
    """
    return ibis.schema(
        [('a', 'polygon'), ('b', 'point'), ('c', 'int8'), ('d', 'double')]
    )


@pytest.fixture(scope='function')
def test_table(con, test_schema):
    """
    Define fixture for test table.

    Yields
    -------
    ibis.expr.types.TableExpr
    """
    with temp_helper(
        con, kind='table', create=True, create_kwargs={'schema': test_schema}
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


def _random_identifier(suffix):
    return '__ibis_test_{}_{}'.format(suffix, util.guid())


@pytest.fixture
def temp_table(con) -> str:
    """Return a temporary table name.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Yields
    ------
    name : string
        Random table name for a temporary usage.
    """
    with temp_helper(
        con, kind='table', create=False, create_kwargs={'schema': test_schema}
    ) as ret_value:
        yield ret_value


@pytest.fixture(scope='session')
def test_data_db() -> str:
    """Return the database name."""
    return OMNISCIDB_DB


@pytest.fixture
def temp_database(con, test_data_db: str) -> str:
    """Create a temporary database.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient
    test_data_db : str

    Yields
    -------
    str
    """
    with temp_helper(con, kind='database', create=True) as ret_value:
        yield ret_value


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
    with temp_helper(con, kind='view') as ret_value:
        yield ret_value


@pytest.fixture(scope='function')
def test_view(con, test_table) -> str:
    """Create a temporary view.

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Yields
    -------
    str
    """
    with temp_helper(
        con, kind='view', create=True, create_kwargs={'expr': test_table}
    ) as ret_value:
        yield ret_value


@pytest.fixture(scope='function')
def test_user(con) -> list:
    """Return a list with temporary user name and password

    Parameters
    ----------
    con : ibis.omniscidb.OmniSciDBClient

    Yields
    -------
    tupple
    """
    with temp_helper(con, kind="user") as name:
        yield (name, "super")
