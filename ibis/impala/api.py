from ibis.impala.client import (ImpalaConnection,  # noqa: F401
                                ImpalaClient,
                                ImpalaDatabase,
                                ImpalaTable)
from ibis.impala.compiler import dialect  # noqa: F401
from ibis.impala.udf import *  # noqa: F401,F403
from ibis.config import options
import ibis.common as com


def compile(expr, params=None):
    """
    Force compilation of expression as though it were an expression depending
    on Impala. Note you can also call expr.compile()

    Returns
    -------
    compiled : string
    """
    from ibis.impala.compiler import to_sql
    return to_sql(expr, dialect.make_context(params=params))


def verify(expr, params=None):
    """
    Determine if expression can be successfully translated to execute on Impala
    """
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(host='localhost', port=21050, database='default', timeout=45,
            use_ssl=False, ca_cert=None, user=None,
            password=None, auth_mechanism='NOSASL',
            kerberos_service_name='impala', pool_size=8, hdfs_client=None):
    """Create an ImpalaClient for use with Ibis.

    Parameters
    ----------
    host : str, optional
        Host name of the impalad or HiveServer2 in Hive
    port : int, optional
        Impala's HiveServer2 port
    database : str, optional
        Default database when obtaining new cursors
    timeout : int, optional
        Connection timeout in seconds when communicating with HiveServer2
    use_ssl : bool, optional
        Use SSL when connecting to HiveServer2
    ca_cert : str, optional
        Local path to 3rd party CA certificate or copy of server certificate
        for self-signed certificates. If SSL is enabled, but this argument is
        ``None``, then certificate validation is skipped.
    user : str, optional
        LDAP user to authenticate
    password : str, optional
        LDAP password to authenticate
    auth_mechanism : str, optional
        {'NOSASL' <- default, 'PLAIN', 'GSSAPI', 'LDAP'}.
        Use NOSASL for non-secured Impala connections.  Use PLAIN for
        non-secured Hive clusters.  Use LDAP for LDAP authenticated
        connections.  Use GSSAPI for Kerberos-secured clusters.
    kerberos_service_name : str, optional
        Specify particular impalad service principal.

    Examples
    --------
    >>> import ibis
    >>> import os
    >>> hdfs_host = os.environ.get('IBIS_TEST_NN_HOST', 'localhost')
    >>> hdfs_port = int(os.environ.get('IBIS_TEST_NN_PORT', 50070))
    >>> impala_host = os.environ.get('IBIS_TEST_IMPALA_HOST', 'localhost')
    >>> impala_port = int(os.environ.get('IBIS_TEST_IMPALA_PORT', 21050))
    >>> hdfs = ibis.hdfs_connect(host=hdfs_host, port=hdfs_port)
    >>> hdfs  # doctest: +ELLIPSIS
    <ibis.filesystems.WebHDFS object at 0x...>
    >>> client = ibis.impala.connect(
    ...     host=impala_host,
    ...     port=impala_port,
    ...     hdfs_client=hdfs,
    ... )
    >>> client  # doctest: +ELLIPSIS
    <ibis.impala.client.ImpalaClient object at 0x...>

    Returns
    -------
    ImpalaClient
    """
    params = {
        'host': host,
        'port': port,
        'database': database,
        'timeout': timeout,
        'use_ssl': use_ssl,
        'ca_cert': ca_cert,
        'user': user,
        'password': password,
        'auth_mechanism': auth_mechanism,
        'kerberos_service_name': kerberos_service_name
    }

    con = ImpalaConnection(pool_size=pool_size, **params)
    try:
        client = ImpalaClient(con, hdfs_client=hdfs_client)
    except Exception:
        con.close()
        raise
    else:
        if options.default_backend is None:
            options.default_backend = client

    return client
