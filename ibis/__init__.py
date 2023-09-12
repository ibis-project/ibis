"""Initialize Ibis module."""
from __future__ import annotations

__version__ = "6.1.0"

from ibis import examples, util
from ibis.backends.base import BaseBackend
from ibis.common.exceptions import IbisError
from ibis.config import options
from ibis.expr import api
from ibis.expr import types as ir
from ibis.expr.api import *  # noqa: F403
from ibis.expr.operations import udf

__all__ = [  # noqa: PLE0604
    "api",
    "examples",
    "ir",
    "udf",
    "util",
    "BaseBackend",
    "IbisError",
    "options",
    *api.__all__,
]

_KNOWN_BACKENDS = ["heavyai"]


def __dir__() -> list[str]:
    """Adds tab completion for ibis backends to the top-level module."""

    out = set(__all__)
    out.update(ep.name for ep in util.backend_entry_points())
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
    entry_points = {ep for ep in util.backend_entry_points() if ep.name == name}

    if not entry_points:
        msg = f"module 'ibis' has no attribute '{name}'. "
        if name in _KNOWN_BACKENDS:
            msg += f"""If you are trying to access the '{name}' backend,
                    try installing it first with `pip install 'ibis-framework[{name}]'`"""
        raise AttributeError(msg)

    if len(entry_points) > 1:
        raise RuntimeError(
            f"{len(entry_points)} packages found for backend '{name}': "
            f"{entry_points}\n"
            "There should be only one, please uninstall the unused packages "
            "and just leave the one that needs to be used."
        )

    import types

    import ibis

    (entry_point,) = entry_points
    try:
        module = entry_point.load()
    except ImportError as exc:
        raise ImportError(
            f"Failed to import the {name} backend due to missing dependencies.\n\n"
            f"You can pip or conda install the {name} backend as follows:\n\n"
            f'  python -m pip install -U "ibis-framework[{name}]"  # pip install\n'
            f"  conda install -c conda-forge ibis-{name}           # or conda install"
        ) from exc
    backend = module.Backend()
    # The first time a backend is loaded, we register its options, and we set
    # it as an attribute of `ibis`, so `__getattr__` is not called again for it
    backend.register_options()

    # We don't want to expose all the methods on an unconnected backend to the user.
    # In lieu of a full redesign, we create a proxy module and add only the methods
    # that are valid to call without a connect call. These are:
    #
    # - connect
    # - compile
    # - has_operation
    # - add_operation
    # - _from_url
    # - _to_sql
    # - _sqlglot_dialect (if defined)
    #
    # We also copy over the docstring from `do_connect` to the proxy `connect`
    # method, since that's where all the backend-specific kwargs are currently
    # documented. This is all admittedly gross, but it works and doesn't
    # require a backend redesign yet.

    def connect(*args, **kwargs):
        return backend.connect(*args, **kwargs)

    connect.__doc__ = backend.do_connect.__doc__
    connect.__wrapped__ = backend.do_connect
    connect.__module__ = f"ibis.{name}"

    proxy = types.ModuleType(f"ibis.{name}")
    setattr(ibis, name, proxy)
    proxy.connect = connect
    proxy.compile = backend.compile
    proxy.has_operation = backend.has_operation
    proxy.add_operation = backend.add_operation
    proxy.name = name
    proxy._from_url = backend._from_url
    proxy._to_sql = backend._to_sql
    if (dialect := getattr(backend, "_sqlglot_dialect", None)) is not None:
        proxy._sqlglot_dialect = dialect
    # Add any additional methods that should be exposed at the top level
    for name in getattr(backend, "_top_level_methods", ()):
        setattr(proxy, name, getattr(backend, name))

    return proxy
