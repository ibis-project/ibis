"""Initialize Ibis module."""
import importlib
import pkg_resources

import ibis.config_init  # noqa: F401
import ibis.expr.api as api  # noqa: F401
import ibis.expr.types as ir  # noqa: F401

# pandas backend is mandatory
import ibis.pandas.api as pandas  # noqa: F401
import ibis.util as util  # noqa: F401
from ibis.common.exceptions import IbisError
from ibis.config import options  # noqa: F401
from ibis.expr.api import *  # noqa: F401,F403
from ibis.filesystems import HDFS, WebHDFS  # noqa: F401

from ._version import get_versions  # noqa: E402


def hdfs_connect(
    host='localhost',
    port=50070,
    protocol='webhdfs',
    use_https='default',
    auth_mechanism='NOSASL',
    verify=True,
    session=None,
    **kwds,
):
    """Connect to HDFS.

    Parameters
    ----------
    host : str
        Host name of the HDFS NameNode
    port : int
        NameNode's WebHDFS port
    protocol : str,
        The protocol used to communicate with HDFS. The only valid value is
        ``'webhdfs'``.
    use_https : bool
        Connect to WebHDFS with HTTPS, otherwise plain HTTP. For secure
        authentication, the default for this is True, otherwise False.
    auth_mechanism : str
        Set to NOSASL or PLAIN for non-secure clusters.
        Set to GSSAPI or LDAP for Kerberos-secured clusters.
    verify : bool
        Set to :data:`False` to turn off verifying SSL certificates.
    session : Optional[requests.Session]
        A custom :class:`requests.Session` object.

    Notes
    -----
    Other keywords are forwarded to HDFS library classes.

    Returns
    -------
    WebHDFS

    """
    import requests

    if session is None:
        session = requests.Session()
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
                "requests-kerberos` or `pip install hdfs[kerberos]`."
            )
        from hdfs.ext.kerberos import KerberosClient

        # note SSL
        url = '{0}://{1}:{2}'.format(prefix, host, port)
        kwds.setdefault('mutual_auth', 'OPTIONAL')
        hdfs_client = KerberosClient(url, session=session, **kwds)
    else:
        if use_https == 'default':
            prefix = 'http'
        else:
            prefix = 'https' if use_https else 'http'
        from hdfs.client import InsecureClient

        url = '{}://{}:{}'.format(prefix, host, port)
        hdfs_client = InsecureClient(url, session=session, **kwds)
    return WebHDFS(hdfs_client)


__version__ = get_versions()['version']
del get_versions


def __getattr__(name):
    """
    Load backends as `ibis` module attributes.

    When `ibis.sqlite` is called, this function is executed with `name=sqlite`.
    Ibis backends are expected to be defined as `entry_points` in the
    `setup.py` file of the Ibis project, or of third-party backends.

    If a backend is not found in the entry point registry, and `ImportError` is
    raised.
    """
    try:
        entry_point = next(
            pkg_resources.iter_entry_points('ibis.backends', name)
        )
    except StopIteration:
        raise ImportError(
            f'"{name}" was assumed to be a backend, but it was '
            'not found. You may have to install it with `pip '
            f'install ibis-{name}`.'
        )
    module = importlib.import_module(entry_point.module_name)
    if not hasattr(module, 'connect'):
        raise RuntimeError(f'"{entry_point.module_name}" module does not seem '
                           'to be an Ibis backend.'
                           f' {module} - {module.__file__} - {dir(module)}')
    return module
