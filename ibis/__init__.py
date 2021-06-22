"""Initialize Ibis module."""
import warnings

import pkg_resources

# Converting an Ibis schema to a pandas DataFrame requires registering
# some type conversions that are currently registered in the pandas backend
import ibis.backends.pandas
import ibis.config
import ibis.expr.types as ir
from ibis import util
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
with ibis.config.config_prefix('context_adjustment'):
    ibis.config.register_option(
        'time_col',
        'time',
        'Name of the timestamp col for execution with a timecontext'
        'See ibis.expr.timecontext for details.',
        validator=ibis.config.is_str,
    )
with ibis.config.config_prefix('sql'):
    ibis.config.register_option(
        'default_limit',
        10_000,
        'Number of rows to be retrieved for an unlimited table expression',
    )

__version__ = get_versions()['version']
del get_versions


def __getattr__(name: str):
    entry_points = list(
        pkg_resources.iter_entry_points(group='ibis.backends', name=name)
    )
    if len(entry_points) == 0:
        raise AttributeError(
            f"module 'ibis' has no attribute '{name}'. "
            f"If you are trying to access the '{name}' backend, "
            f"try installing it first with `pip install ibis-{name}`"
        )
    elif len(entry_points) > 1:
        warnings.warn(
            f"More than one entrypoint found for backend '{name}', "
            f"using {entry_points[0].module_name}"
        )

    backend = entry_points[0].resolve().Backend()

    # The first time a backend is loaded, we register its options, and we set
    # it as an attribute, so `__getattr__` is not loaded again.
    with ibis.config.config_prefix(name):
        backend.register_options()

    setattr(ibis, name, backend)
    return backend
