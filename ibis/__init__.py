import sys

import ibis.config_init  # noqa: F401
import ibis.util as util  # noqa: F401
import ibis.expr.api as api  # noqa: F401
import ibis.expr.types as ir  # noqa: F401

from ibis.config import options  # noqa: F401
from ibis.common import IbisError
from ibis.compat import suppress
from ibis.filesystems import HDFS, WebHDFS  # noqa: F401

# __all__ is defined
from ibis.expr.api import *  # noqa: F401,F403

# pandas backend is mandatory
import ibis.pandas.api as pandas  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[csv]
    import ibis.file.csv as csv  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[parquet]
    import ibis.file.parquet as parquet  # noqa: F401

with suppress(ImportError):
    # pip install  ibis-framework[hdf5]
    import ibis.file.hdf5 as hdf5  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[impala]
    import ibis.impala.api as impala  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[sqlite]
    import ibis.sql.sqlite.api as sqlite  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[postgres]
    import ibis.sql.postgres.api as postgres  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[mysql]
    import ibis.sql.mysql.api as mysql  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[clickhouse]
    import ibis.clickhouse.api as clickhouse  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[bigquery]
    import ibis.bigquery.api as bigquery  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[mapd]
    if sys.version_info.major >= 3:
        import ibis.mapd.api as mapd  # noqa: F401


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
    if auth_mechanism in ('GSSAPI', 'LDAP'):
        if use_https == 'default':
            prefix = 'https'
        else:
            prefix = 'https' if use_https else 'http'
        try:
            import requests_kerberos  # noqa: F401
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
        url = '{}://{}:{}'.format(prefix, host, port)
        hdfs_client = InsecureClient(url, **kwds)
    return WebHDFS(hdfs_client)


from ._version import get_versions  # noqa: E402
__version__ = get_versions()['version']
del get_versions
