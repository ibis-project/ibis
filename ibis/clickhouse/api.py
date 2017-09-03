from ibis.clickhouse.client import ClickhouseClient
# ClickhouseConnection, ClickhouseDatabase, ClickhouseTable

from ibis.config import options
import ibis.common as com


def compile(expr):
    """
    Force compilation of expression as though it were an expression depending
    on Clickhouse. Note you can also call expr.compile()

    Returns
    -------
    compiled : string
    """
    from .compiler import to_sql
    return to_sql(expr)


def verify(expr):
    """
    Determine if expression can be successfully translated to execute on
    Clickhouse
    """
    try:
        compile(expr)
        return True
    except com.TranslationError:
        return False


def connect(host='localhost', port=9000, database='default', user='default',
            password='', client_name='ibis', compression=False):
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
        Weather or not to use compression. Default is False.
        Possible choices: lz4, lz4hc, quicklz, zstd
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
