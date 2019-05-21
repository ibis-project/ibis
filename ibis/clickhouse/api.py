import ibis.common as com

from ibis.config import options
from ibis.clickhouse.client import ClickhouseClient
from ibis.clickhouse.compiler import dialect


__all__ = 'compile', 'verify', 'connect', 'dialect'


try:
    import lz4  # noqa: F401
    _default_compression = 'lz4'
except ImportError:
    _default_compression = False


def compile(expr, params=None):
    """
    Force compilation of expression as though it were an expression depending
    on Clickhouse. Note you can also call expr.compile()

    Returns
    -------
    compiled : string
    """
    from ibis.clickhouse.compiler import to_sql
    return to_sql(expr, dialect.make_context(params=params))


def verify(expr, params=None):
    """
    Determine if expression can be successfully translated to execute on
    Clickhouse
    """
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(host='localhost', port=9000, database='default', user='default',
            password='', client_name='ibis', compression=_default_compression):
    """Create an ClickhouseClient for use with Ibis.

    Parameters
    ----------
    host : str, optional
        Host name of the clickhouse server
    port : int, optional
        Clickhouse server's  port
    database : str, optional
        Default database when executing queries
    user : str, optional
        User to authenticate with
    password : str, optional
        Password to authenticate with
    client_name: str, optional
        This will appear in clickhouse server logs
    compression: str, optional
        Weather or not to use compression.
        Default is lz4 if installed else False.
        Possible choices: lz4, lz4hc, quicklz, zstd, True, False
        True is equivalent to 'lz4'.

    Examples
    --------
    >>> import ibis
    >>> import os
    >>> clickhouse_host = os.environ.get('IBIS_TEST_CLICKHOUSE_HOST',
    ...                                  'localhost')
    >>> clickhouse_port = int(os.environ.get('IBIS_TEST_CLICKHOUSE_PORT',
    ...                                      9000))
    >>> client = ibis.clickhouse.connect(
    ...     host=clickhouse_host,
    ...     port=clickhouse_port
    ... )
    >>> client  # doctest: +ELLIPSIS
    <ibis.clickhouse.client.ClickhouseClient object at 0x...>

    Returns
    -------
    ClickhouseClient
    """
    client = ClickhouseClient(host, port=port, database=database, user=user,
                              password=password, client_name=client_name,
                              compression=compression)
    if options.default_backend is None:
        options.default_backend = client

    return client
