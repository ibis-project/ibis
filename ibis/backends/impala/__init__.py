"""Impala backend"""
import ibis.config
from ibis.backends.base import BaseBackend

# these objects are exposed in the public API and are not used in the module
from .client import (  # noqa: F401
    ImpalaClient,
    ImpalaConnection,
    ImpalaDatabase,
    ImpalaDatabaseTable,
    ImpalaQuery,
    ImpalaTable,
)
from .compiler import ImpalaDialect, ImpalaQueryBuilder
from .hdfs import HDFS, WebHDFS, hdfs_connect
from .udf import *  # noqa: F401,F403


class Backend(BaseBackend):
    name = 'impala'
    builder = ImpalaQueryBuilder
    dialect = ImpalaDialect
    database_class = ImpalaDatabase
    query_class = ImpalaQuery
    table_class = ImpalaDatabaseTable
    table_expr_class = ImpalaTable
    HDFS = HDFS
    WebHDFS = WebHDFS

    def hdfs_connect(self, *args, **kwargs):
        return hdfs_connect(*args, **kwargs)

    def connect(
        self,
        host='localhost',
        port=21050,
        database='default',
        timeout=45,
        use_ssl=False,
        ca_cert=None,
        user=None,
        password=None,
        auth_mechanism='NOSASL',
        kerberos_service_name='impala',
        pool_size=8,
        hdfs_client=None,
    ):
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
            Local path to 3rd party CA certificate or copy of server
            certificate for self-signed certificates. If SSL is enabled, but
            this argument is ``None``, then certificate validation is skipped.
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
        >>> hdfs = ibis.impala.hdfs_connect(host=hdfs_host, port=hdfs_port)
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
            'kerberos_service_name': kerberos_service_name,
        }

        con = ImpalaConnection(pool_size=pool_size, **params)
        try:
            client = ImpalaClient(
                backend=self, con=con, hdfs_client=hdfs_client
            )
        except Exception:
            con.close()
            raise
        return client

    def register_options(self):
        ibis.config.register_option(
            'temp_db',
            '__ibis_tmp',
            'Database to use for temporary tables, views. functions, etc.',
        )
        ibis.config.register_option(
            'temp_hdfs_path',
            '/tmp/ibis',
            'HDFS path for storage of temporary data',
        )
