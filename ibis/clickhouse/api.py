# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ibis.clickhouse.client import ClickhouseClient  # noqa
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


def connect(host='localhost', port=9000, database='default', timeout=45,
            ca_cert=None, user='default', client_name='ibis',
            password='', compression=False):
    """Create an ClickhouseClient for use with Ibis.

    Parameters
    ----------
    host : str, optional
        Host name of the clickhoused or HiveServer2 in Hive
    port : int, optional
        Clickhouse's HiveServer2 port
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
    ...     port=clickhouse_port,
    ...     hdfs_client=hdfs,
    ... )
    >>> client  # doctest: +ELLIPSIS
    <ibis.clickhouse.client.ClickhouseClient object at 0x...>

    Returns
    -------
    ClickhouseClient
    """
    params = {
        'port': port,
        'database': database,
        'user': user,
        'password': password,
        'client_name': client_name,
        'compression': compression
    }

    try:
        client = ClickhouseClient(host, **params)
    except:
        raise
    else:
        if options.default_backend is None:
            options.default_backend = client

    return client
