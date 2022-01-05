"""Initialize Ibis module."""

# Converting an Ibis schema to a pandas DataFrame requires registering
# some type conversions that are currently registered in the pandas backend
import ibis.backends.pandas
import ibis.config
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import BaseBackend
from ibis.common.exceptions import IbisError
from ibis.config import options
from ibis.expr import api
from ibis.expr.api import *  # noqa: F401,F403

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    # TODO: remove this when Python 3.7 support is dropped
    import importlib_metadata

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

try:
    __version__ = importlib_metadata.version(__name__)
except Exception:
    __version__ = importlib_metadata.version("ibis-framework")


def __getattr__(name: str) -> BaseBackend:
    """Load backends in a lazy way with `ibis.<backend-name>`.

    This also registers the backend options.

    Examples
    --------
    >>> import ibis
    >>> con = ibis.sqlite.connect(...)

    When accessing the `sqlite` attribute of the `ibis` module, this function
    is called, and a backend with the `sqlite` name is tried to load from
    the `ibis.backends` entrypoints. If successful, the `ibis.sqlite`
    attribute is "cached", so this function is only called the first time.
    """
    entry_points = {
        entry_point
        for entry_point in importlib_metadata.entry_points()["ibis.backends"]
        if name == entry_point.name
    }

    if not entry_points:
        raise AttributeError(
            f"module 'ibis' has no attribute '{name}'. "
            f"If you are trying to access the '{name}' backend, "
            f"try installing it first with `pip install ibis-{name}`"
        )

    if len(entry_points) > 1:
        raise RuntimeError(
            f"{len(entry_points)} packages found for backend '{name}': "
            f"{entry_points}\n"
            "There should be only one, please uninstall the unused packages "
            "and just leave the one that needs to be used."
        )

    (entry_point,) = entry_points
    module = entry_point.load()
    backend = module.Backend()

    # The first time a backend is loaded, we register its options, and we set
    # it as an attribute of `ibis`, so `__getattr__` is not called again for it
    with ibis.config.config_prefix(name):
        backend.register_options()

    setattr(ibis, name, backend)
    return backend
