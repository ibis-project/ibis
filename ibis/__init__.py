"""Initialize Ibis module."""
import pkg_resources

import ibis.config
import ibis.expr.types as ir
from ibis import util

# pandas backend is mandatory
from ibis.backends import pandas  # noqa: F401
from ibis.common.exceptions import IbisError
from ibis.config import options
from ibis.expr import api
from ibis.expr.api import *  # noqa: F401,F403

from ._version import get_versions  # noqa: E402

__all__ = ['api', 'ir', 'util', 'IbisError', 'options']
__all__ += api.__all__


ibis.config.register_option(
    'interactive', False, validator=ibis.config.is_bool
)
ibis.config.register_option('verbose', False, validator=ibis.config.is_bool)
ibis.config.register_option('verbose_log', None)
ibis.config.register_option(
    'graphviz_repr',
    True,
    """\
Whether to render expressions as GraphViz PNGs when repr-ing in a Jupyter
notebook.
""",
    validator=ibis.config.is_bool,
)
ibis.config.register_option('default_backend', None)

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
