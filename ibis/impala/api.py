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

from ibis.impala.client import (ImpalaConnection,  # noqa
                                ImpalaClient,
                                ImpalaDatabase,
                                ImpalaTable)
from ibis.impala.udf import *  # noqa
from ibis.impala.madlib import MADLibAPI  # noqa
from ibis.config import options
import ibis.common as com


def compile(expr):
    """
    Force compilation of expression as though it were an expression depending
    on Impala. Note you can also call expr.compile()

    Returns
    -------
    compiled : string
    """
    from .compiler import to_sql
    return to_sql(expr)


def verify(expr):
    """
    Determine if expression can be successfully translated to execute on Impala
    """
    try:
        compile(expr)
        return True
    except com.TranslationError:
        return False


def connect(host='localhost', port=21050, database='default', timeout=45,
            use_ssl=False, ca_cert=None, user=None,
            password=None, auth_mechanism='NOSASL',
            kerberos_service_name='impala', pool_size=8, hdfs_client=None):
    """
    Create an ImpalaClient for use with Ibis.

    Parameters
    ----------
    host : string, Host name of the impalad or HiveServer2 in Hive
    port : int, Defaults to 21050 (Impala's HiveServer2)
    database : string, Default database when obtaining new cursors
    timeout : int, Connection timeout (seconds) when communicating with
        HiveServer2
    use_ssl : boolean, Use SSL when connecting to HiveServer2
    ca_cert : string, Local path to 3rd party CA certificate or copy of server
        certificate for self-signed certificates. If SSL is enabled, but this
        argument is None, then certificate validation is skipped.
    user : string, LDAP user to authenticate
    password : string, LDAP password to authenticate
    auth_mechanism : string, {'NOSASL' <- default, 'PLAIN', 'GSSAPI', 'LDAP'}.
        Use NOSASL for non-secured Impala connections.  Use PLAIN for
        non-secured Hive clusters.  Use LDAP for LDAP authenticated
        connections.  Use GSSAPI for Kerberos-secured clusters.
    kerberos_service_name : string, Specify particular impalad service
        principal.

    Examples
    --------
    >>> hdfs = ibis.hdfs_connect(**hdfs_params)
    >>> client = ibis.impala.connect(hdfs_client=hdfs, **impala_params)

    Returns
    -------
    con : ImpalaClient
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
    client = ImpalaClient(con, hdfs_client=hdfs_client)

    if options.default_backend is None:
        options.default_backend = client

    return client
