# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# flake8: noqa

__version__ = '0.5.0'

from ibis.filesystems import HDFS, WebHDFS
from ibis.common import IbisError

import ibis.expr.api as api
import ibis.expr.types as ir

# __all__ is defined
from ibis.expr.api import *

import ibis.impala.api as impala
import ibis.sql.sqlite.api as sqlite
import ibis.sql.postgres.api as postgres

import ibis.config_init
from ibis.config import options
import ibis.util as util


def hdfs_connect(host='localhost', port=50070, protocol='webhdfs',
                 use_https='default', auth_mechanism='NOSASL',
                 verify=True, **kwds):
    """
    Connect to HDFS

    Parameters
    ----------
    host : string, Host name of the HDFS NameNode
    port : int, NameNode's WebHDFS port (default 50070)
    protocol : {'webhdfs'}
    use_https : boolean, default 'default'
        Connect to WebHDFS with HTTPS, otherwise plain HTTP. For secure
        authentication, the default for this is True, otherwise False
    auth_mechanism : string, Set to NOSASL or PLAIN for non-secure clusters.
        Set to GSSAPI or LDAP for Kerberos-secured clusters.
    verify : boolean, Set to False to turn off verifying SSL certificates.
        (default True)

    Other keywords are forwarded to hdfs library classes

    Returns
    -------
    client : WebHDFS
    """
    import requests
    session = kwds.setdefault('session', requests.Session())
    session.verify = verify
    if auth_mechanism in ['GSSAPI', 'LDAP']:
        if use_https == 'default':
            prefix = 'https'
        else:
            prefix = 'https' if use_https else 'http'
        try:
            import requests_kerberos
        except ImportError:
            raise IbisError(
                "Unable to import requests-kerberos, which is required for "
                "Kerberos HDFS support. Install it by executing `pip install "
                "requests-kerberos` or `pip install hdfs[kerberos]`.")
        from hdfs.ext.kerberos import KerberosClient
        # note SSL
        url = '{0}://{1}:{2}'.format(prefix, host, port)
        kwds.setdefault('mutual_auth', 'OPTIONAL')
        hdfs_client = KerberosClient(url, **kwds)
    else:
        if use_https == 'default':
            prefix = 'http'
        else:
            prefix = 'https' if use_https else 'http'
        from hdfs.client import InsecureClient
        url = '{0}://{1}:{2}'.format(prefix, host, port)
        hdfs_client = InsecureClient(url, **kwds)
    return WebHDFS(hdfs_client)

def test(impala=False):
    import pytest
    import ibis
    import os

    ibis_dir, _ = os.path.split(ibis.__file__)

    args = ['--pyargs', ibis_dir]
    if impala:
        args.append('--impala')
    pytest.main(args)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
