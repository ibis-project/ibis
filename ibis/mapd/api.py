from ibis.config import options
from ibis.mapd.client import MapDClient
from ibis.mapd.compiler import dialect

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
    uri: str=None, user: str=None, password: str=None, host: str=None,
    port: int=9091, dbname: str=None, protocol: str='binary',
    execution_type: int=3
):
    """Create a MapDClient for use with Ibis

    Parameters could be

    :param uri: str
    :param user: str
    :param password: str
    :param host: str
    :param port: int
    :param dbname: str
    :param protocol: str
    :param execution_type: int
    Returns
    -------
    MapDClient

    """
    client = MapDClient(
        uri=uri, user=user, password=password, host=host,
        port=port, dbname=dbname, protocol=protocol,
        execution_type=execution_type
    )

    if options.default_backend is None:
        options.default_backend = client

    return client
