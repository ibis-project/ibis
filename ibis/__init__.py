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


print('******************* LOADING BACKENDS ******************')
for entry_point in pkg_resources.iter_entry_points(
    group='ibis.backends', name=None
):
    print(f'{entry_point.name}')
    try:
        setattr(ibis, entry_point.name, entry_point.resolve())
    except ImportError as exc:
        print(f'- failed: {exc}')
        pass
