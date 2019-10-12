"""OmniSciDB API module."""

import ibis
import ibis.common.exceptions as com
import ibis.config as cf
from ibis.config import options
from ibis.omniscidb.client import EXECUTION_TYPE_CURSOR, OmniSciDBClient
from ibis.omniscidb.compiler import compiles, dialect, rewrites  # noqa: F401


def compile(expr: ibis.Expr, params=None):
    """Compile a given expression.

    Note you can also call expr.compile().

    Parameters
    ----------
    expr : ibis.Expr
    params : dict

    Returns
    -------
    compiled : string
    """
    from ibis.omniscidb.compiler import to_sql

    return to_sql(expr, dialect.make_context(params=params))


def verify(expr: ibis.Expr, params: dict = None) -> bool:
    """Check if a given expression can be successfully translated.

    Parameters
    ----------
    expr : ibis.Expr
    params : dict, optional

    Returns
    -------
    bool
    """
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(
    uri=None,
    user=None,
    password=None,
    host=None,
    port=6274,
    database=None,
    protocol='binary',
    session_id=None,
    execution_type=EXECUTION_TYPE_CURSOR,
):
    """Create a client for OmniSciDB backend.

    Parameters
    ----------
    uri : str
    user : str
    password : str
    host : str
    port : int
    database : str
    protocol : str
    session_id : str
    execution_type : int

    Returns
    -------
    OmniSciDBClient
    """
    client = OmniSciDBClient(
        uri=uri,
        user=user,
        password=password,
        host=host,
        port=port,
        database=database,
        protocol=protocol,
        session_id=session_id,
        execution_type=execution_type,
    )

    if options.default_backend is None:
        options.default_backend = client

    with cf.config_prefix('sql'):
        k = 'default_limit'
        cf.set_option(k, None)

    return client
