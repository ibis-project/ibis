"""Initialize Ibis module."""
import pkg_resources

import ibis.config_init  # noqa: F401
import ibis.expr.api as api  # noqa: F401
import ibis.expr.types as ir  # noqa: F401
import ibis.util as util  # noqa: F401
from ibis.common.exceptions import IbisError  # noqa: F401
from ibis.config import options  # noqa: F401
from ibis.expr.api import *  # noqa: F401,F403
from ibis.filesystems import HDFS, WebHDFS, hdfs_connect  # noqa: F401

from ._version import get_versions  # noqa: E402


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
    return entry_point.resolve()
