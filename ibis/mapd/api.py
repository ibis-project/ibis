from ibis.config import options
from ibis.mapd.compiler import dialect, compiles, rewrites  # noqa: F401
from ibis.mapd.client import MapDClient, EXECUTION_TYPE_CURSOR

import ibis.common as com


def compile(expr, params=None):
    """
    Force compilation of expression as though it were an expression depending
    on MapD. Note you can also call expr.compile()

    Returns
    -------
    compiled : string
    """
    from ibis.mapd.compiler import to_sql
    return to_sql(expr, dialect.make_context(params=params))


def verify(expr, params=None):
    """
    Determine if expression can be successfully translated to execute on
    MapD
    """
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(
    uri=None, user=None, password=None, host=None,
    port=9091, database=None, protocol='binary',
    execution_type=EXECUTION_TYPE_CURSOR
):
    """Create a MapDClient for use with Ibis

    Parameters could be

    :param uri: str
    :param user: str
    :param password: str
    :param host: str
    :param port: int
    :param database: str
    :param protocol: str
    :param execution_type: int
    Returns
    -------
    MapDClient

    """
    client = MapDClient(
        uri=uri, user=user, password=password, host=host,
        port=port, database=database, protocol=protocol,
        execution_type=execution_type
    )

    if options.default_backend is None:
        options.default_backend = client

    return client
