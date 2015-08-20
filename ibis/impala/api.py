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


def connect(host='localhost', port=21050, protocol='hiveserver2',
            database='default', timeout=45, use_ssl=False, ca_cert=None,
            use_ldap=False, ldap_user=None, ldap_password=None,
            use_kerberos=False, kerberos_service_name='impala',
            pool_size=8):
    """
    Create an Impala Client for use with Ibis

    Parameters
    ----------
    host : host name
    port : int, default 21050 (HiveServer 2)
    protocol : {'hiveserver2', 'beeswax'}
    database :
    timeout :
    use_ssl :
    ca_cert :
    use_ldap : boolean, default False
    ldap_user :
    ldap_password :
    use_kerberos : boolean, default False
    kerberos_service_name : string, default 'impala'

    Returns
    -------
    con : ImpalaConnection
    """
    params = {
        'host': host,
        'port': port,
        'protocol': protocol,
        'database': database,
        'timeout': timeout,
        'use_ssl': use_ssl,
        'ca_cert': ca_cert,
        'use_ldap': use_ldap,
        'ldap_user': ldap_user,
        'ldap_password': ldap_password,
        'use_kerberos': use_kerberos,
        'kerberos_service_name': kerberos_service_name
    }

    return ImpalaConnection(pool_size=pool_size, **params)
