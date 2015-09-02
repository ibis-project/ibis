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

from ibis.impala.client import (ImpalaConnection, ImpalaClient,  # noqa
                                Database, ImpalaTable)
from ibis.impala.udf import *  # noqa
from ibis.impala.madlib import MADLibAPI  # noqa


def connect(host='localhost', port=21050, database='default', timeout=45,
            use_ssl=False, ca_cert=None, user=None, password=None,
            auth_mechanism='NOSASL', kerberos_service_name='impala',
            pool_size=8):
    """
    Create an Impala Client for use with Ibis

    Parameters
    ----------
    host : host name
    port : int, default 21050 (HiveServer 2)
    database :
    timeout :
    use_ssl : boolean
    ca_cert :
    user :
    password :
    auth_mechanism : {'NOSASL' <- default, 'PLAIN', 'GSSAPI', 'LDAP'}
    kerberos_service_name : string, default 'impala'

    Returns
    -------
    con : ImpalaConnection
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

    return ImpalaConnection(pool_size=pool_size, **params)
