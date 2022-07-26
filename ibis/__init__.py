"""Initialize Ibis module."""
from __future__ import annotations

import importlib.metadata as _importlib_metadata

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

__all__ = ['api', 'ir', 'util', 'BaseBackend', 'IbisError', 'options']
__all__ += api.__all__

__version__ = "3.1.0"


def _get_backend_entrypoints() -> list[_importlib_metadata.EntryPoint]:
    """Get the list of installed `ibis.backend` entrypoints"""
    import sys

    if sys.version_info < (3, 10):
        return list(_importlib_metadata.entry_points()['ibis.backends'])
    else:
        return list(_importlib_metadata.entry_points(group="ibis.backends"))


def __dir__() -> list[str]:
    """Adds tab completion for ibis backends to the top-level module"""
    out = set(__all__)
    out.update(ep.name for ep in _get_backend_entrypoints())
    return sorted(out)


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
    entry_points = {ep for ep in _get_backend_entrypoints() if ep.name == name}

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
    backend.register_options()

    setattr(ibis, name, backend)
    return backend
