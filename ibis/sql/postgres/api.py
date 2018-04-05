from ibis.sql.alchemy import to_sqlalchemy

from ibis.sql.postgres.client import PostgreSQLClient
from ibis.sql.postgres.compiler import (  # noqa: F401, E501
    compiles,
    rewrites,
    dialect
)


def compile(expr, params=None):
    """Compile an ibis expression to the PostgreSQL target.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        The ibis expression to compile
    params : dict or None
        ``dict`` mapping :class:`ibis.expr.types.ScalarParameter` objects to
        values

    Returns
    -------
    sqlalchemy_expression : sqlalchemy.sql.expression.ClauseElement

    Examples
    --------
    >>> import os
    >>> import getpass
    >>> host = os.environ.get('IBIS_TEST_POSTGRES_HOST', 'localhost')
    >>> user = os.environ.get('IBIS_TEST_POSTGRES_USER', getpass.getuser())
    >>> password = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD')
    >>> database = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
    ...                           'ibis_testing')
    >>> con = connect(
    ...     database=database,
    ...     host=host,
    ...     user=user,
    ...     password=password
    ... )
    >>> t = con.table('functional_alltypes')
    >>> expr = t.double_col + 1
    >>> sqla = compile(expr)
    >>> print(str(sqla))  # doctest: +NORMALIZE_WHITESPACE
    SELECT t0.double_col + %(param_1)s AS tmp
    FROM functional_alltypes AS t0
    """
    return to_sqlalchemy(expr, dialect.make_context(params=params))


def connect(
    host='localhost',
    user=None,
    password=None,
    port=5432,
    database=None,
    url=None,
    driver='psycopg2'
):
    """Create an Ibis client located at `user`:`password`@`host`:`port`
    connected to a PostgreSQL database named `database`.

    Parameters
    ----------
    host : string, default 'localhost'
    user : string, default None
    password : string, default None
    port : string or integer, default 5432
    database : string, default None
    url : string, default None
        Complete SQLAlchemy connection string. If passed, the other connection
        arguments are ignored.
    driver : string, default 'psycopg2'

    Returns
    -------
    PostgreSQLClient

    Examples
    --------
    >>> import os
    >>> import getpass
    >>> host = os.environ.get('IBIS_TEST_POSTGRES_HOST', 'localhost')
    >>> user = os.environ.get('IBIS_TEST_POSTGRES_USER', getpass.getuser())
    >>> password = os.environ.get('IBIS_TEST_POSTGRES_PASSWORD')
    >>> database = os.environ.get('IBIS_TEST_POSTGRES_DATABASE',
    ...                           'ibis_testing')
    >>> con = connect(
    ...     database=database,
    ...     host=host,
    ...     user=user,
    ...     password=password
    ... )
    >>> con.list_tables()  # doctest: +ELLIPSIS
    [...]
    >>> t = con.table('functional_alltypes')
    >>> t
    PostgreSQLTable[table]
      name: functional_alltypes
      schema:
        index : int64
        Unnamed: 0 : int64
        id : int32
        bool_col : boolean
        tinyint_col : int16
        smallint_col : int16
        int_col : int32
        bigint_col : int64
        float_col : float
        double_col : double
        date_string_col : string
        string_col : string
        timestamp_col : timestamp
        year : int32
        month : int32
    """
    return PostgreSQLClient(
        host=host,
        user=user,
        password=password,
        port=port,
        database=database,
        url=url,
        driver=driver,
    )
