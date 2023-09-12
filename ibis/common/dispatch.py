from __future__ import annotations

import abc
import functools
import re
from collections import defaultdict

from ibis.util import import_object


def normalize(r: str | re.Pattern):
    """Normalize a expression by wrapping it with `'^'` and `'$'`.

    Parameters
    ----------
    r
        The pattern to normalize.

    Returns
    -------
    Pattern
        The compiled regex.
    """
    r = getattr(r, "pattern", r)
    return re.compile("^" + r.lstrip("^").rstrip("$") + "$")


def lazy_singledispatch(func):
    """A `singledispatch` implementation that supports lazily registering implementations."""

    lookup = {object: func}
    abc_lookup = {}
    lazy_lookup = defaultdict(dict)

    def register(cls, func=None):
        """Registers a new implementation for arguments of type `cls`."""

        def inner(func):
            if isinstance(cls, tuple):
                for t in cls:
                    register(t, func)
            elif isinstance(cls, abc.ABCMeta):
                abc_lookup[cls] = func
            elif isinstance(cls, str):
                module = cls.split(".", 1)[0]
                lazy_lookup[module][cls] = func
            else:
                lookup[cls] = func
            return func

        return inner if func is None else inner(func)

    def dispatch(cls):
        """Return the implementation for the given `cls`."""
        for cls2 in cls.__mro__:
            # 1. Check for a concrete implementation
            try:
                impl = lookup[cls2]
            except KeyError:
                pass
            else:
                if cls is not cls2:
                    # Cache implementation
                    lookup[cls] = impl
                return impl
            # 2. Check lazy implementations
            module = cls2.__module__.split(".", 1)[0]
            if lazy := lazy_lookup.get(module):
                # Import all lazy implementations first before registering
                # (which should never fail), to ensure an error anywhere
                # doesn't result in a half-registered state.
                new = {import_object(name): func for name, func in lazy.items()}
                lookup.update(new)
                # drop lazy implementations, idempotent for thread safety
                lazy_lookup.pop(module, None)
                return dispatch(cls)
            # 3. Check for abcs
            for abc_cls, impl in abc_lookup.items():
                if issubclass(cls, abc_cls):
                    lookup[cls] = impl
                    return impl
        # Can never get here, since a base `object` implementation is
        # always registered
        raise AssertionError("should never get here")  # pragma: no cover

    @functools.wraps(func)
    def call(arg, *args, **kwargs):
        return dispatch(type(arg))(arg, *args, **kwargs)

    call.dispatch = dispatch
    call.register = register

    return call
