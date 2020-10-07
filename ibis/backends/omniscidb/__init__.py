"""OmniSciDB backend."""
from typing import Optional

import ibis
import ibis.common.exceptions as com
import ibis.config as cf
from ibis.config import options

from .client import OmniSciDBClient  # noqa: F401
from .compiler import compiles, dialect, rewrites  # noqa: F401

__all__ = 'compile', 'verify', 'connect'


def compile(expr: ibis.Expr, params=None):
    """Compile a given expression.

    Note you can also call expr.compile().

    Parameters
    ----------
    expr : ibis.Expr
    params : dict

    Returns
    -------
    compiled : string
    """
    from .compiler import to_sql

    return to_sql(expr, dialect.make_context(params=params))


def verify(expr: ibis.Expr, params: dict = None) -> bool:
    """Check if a given expression can be successfully translated.

    Parameters
    ----------
    expr : ibis.Expr
    params : dict, optional

    Returns
    -------
    bool
    """
    try:
        compile(expr, params=params)
        return True
    except com.TranslationError:
        return False


def connect(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = 6274,
    database: Optional[str] = None,
    protocol: str = 'binary',
    session_id: Optional[str] = None,
    ipc: Optional[bool] = None,
    gpu_device: Optional[int] = None,
):
    """Create a client for OmniSciDB backend.

    Parameters
    ----------
    uri : str, optional
    user : str, optional
    password : str, optional
    host : str, optional
    port : int, default 6274
    database : str, optional
    protocol : {'binary', 'http', 'https'}, default 'binary'
    session_id: str, optional
    ipc : bool, optional
      Enable Inter Process Communication (IPC) execution type.
      `ipc` default value is False when `gpu_device` is None, otherwise
      its default value is True.
    gpu_device : int, optional
      GPU Device ID.

    Returns
    -------
    OmniSciDBClient
    """
    client = OmniSciDBClient(
        uri=uri,
        user=user,
        password=password,
        host=host,
        port=port,
        database=database,
        protocol=protocol,
        session_id=session_id,
        ipc=ipc,
        gpu_device=gpu_device,
    )

    if options.default_backend is None:
        options.default_backend = client

    with cf.config_prefix('sql'):
        k = 'default_limit'
        cf.set_option(k, None)

    return client
