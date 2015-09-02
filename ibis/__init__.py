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

import ibis.config_init
from ibis.config import options
import ibis.util as util


# Deprecated
impala_connect = util.deprecate(impala.connect,
                                'impala_connect is deprecated, use'
                                ' ibis.impala.connect instead')


def make_client(db, hdfs_client=None):
    """
    Create an Ibis client from a database connection and optional additional
    connections (like HDFS)

    Parameters
    ----------
    db : Connection
      e.g. produced by ibis.impala.connect
    hdfs_client : ibis HDFS client

    Examples
    --------
    >>> con = ibis.impala.connect(**impala_params)
    >>> hdfs = ibis.hdfs_connect(**hdfs_params)
    >>> client = ibis.make_client(con, hdfs_client=hdfs)

    Returns
    -------
    client : IbisClient
    """
    client = impala.ImpalaClient(db, hdfs_client=hdfs_client)

    if options.default_backend is None:
        options.default_backend = client

    return client


def hdfs_connect(host='localhost', port=50070, protocol='webhdfs',
                 auth_mechanism='NOSASL', verify=True, **kwds):
    """
    Connect to HDFS

    Parameters
    ----------
    host : string
    port : int, default 50070 (webhdfs default)
    protocol : {'webhdfs'}
    auth_mechanism : {'NOSASL' <- default, 'GSSAPI', 'LDAP', 'PLAIN'}
    verify : boolean, default False
        Set to False to turn off verifying SSL certificates

    Other keywords are forwarded to hdfs library classes

    Returns
    -------
    client : ibis HDFS client
    """
    import requests
    session = kwds.setdefault('session', requests.Session())
    session.verify = verify
    if auth_mechanism in ['GSSAPI', 'LDAP']:
        try:
            import requests_kerberos
        except ImportError:
            raise IbisError(
                "Unable to import requests-kerberos, which is required for "
                "Kerberos HDFS support. Install it by executing `pip install "
                "requests-kerberos` or `pip install hdfs[kerberos]`.")
        from hdfs.ext.kerberos import KerberosClient
        url = 'https://{0}:{1}'.format(host, port) # note SSL
        kwds.setdefault('mutual_auth', 'OPTIONAL')
        hdfs_client = KerberosClient(url, **kwds)
    else:
        from hdfs.client import InsecureClient
        url = 'http://{0}:{1}'.format(host, port)
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
