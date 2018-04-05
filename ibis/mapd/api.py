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


def connect(*args, **kwargs):
    """Create a MapDClient for use with Ibis

    Parameters could be

    host: str
    port: int|str
    database: str
    user: str
    password: str


    Returns
    -------
    MapDClient

    """
    client = MapDClient(*args, **kwargs)

    if options.default_backend is None:
        options.default_backend = client

    return client
