from __future__ import annotations

import ibis.config
from ibis.backends.base.sql import BaseSQLBackend

from .client import ClickhouseClient, ClickhouseTable

_default_compression: str | bool

try:
    import lz4  # noqa: F401

    _default_compression = 'lz4'
except ImportError:
    _default_compression = False


class Backend(BaseSQLBackend):
    name = 'clickhouse'
    client_class = ClickhouseClient
    table_expr_class = ClickhouseTable

    def connect(
        self,
        host='localhost',
        port=9000,
        database='default',
        user='default',
        password='',
        client_name='ibis',
        compression=_default_compression,
    ):
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
        self.client = ClickhouseClient(
            backend=self,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            client_name=client_name,
            compression=compression,
        )
        return self.client

    def register_options(self):
        ibis.config.register_option(
            'temp_db',
            '__ibis_tmp',
            'Database to use for temporary tables, views. functions, etc.',
        )

    @property
    def version(self) -> str:
        self.client.con.connection.force_connect()
        try:
            info = self.client.con.connection.server_info
        except Exception:
            self.client.con.connection.disconnect()
            raise

        return f'{info.version_major}.{info.version_minor}.{info.revision}'

    def list_databases(self, like=None):
        data, schema = self.client.raw_sql('SELECT name FROM system.databases')
        databases = list(data[0])
        return self._filter_with_like(databases, like)
