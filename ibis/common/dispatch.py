from __future__ import annotations

import abc
import functools
import inspect
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


class _MultiDict(dict):
    """A dictionary that allows multiple values for a single key."""

    def __setitem__(self, key, value):
        if key in self:
            self[key].append(value)
        else:
            super().__setitem__(key, [value])


class DispatchedMeta(type):
    """Metaclass that allows multiple implementations of a method to be defined."""

    def __new__(cls, name, bases, dct):
        namespace = {}
        for key, value in dct.items():
            if len(value) == 1:
                # there is just a single attribute so pick that
                namespace[key] = value[0]
            elif all(inspect.isfunction(v) for v in value):
                # multiple functions are defined with the same name, so create
                # a dispatcher function
                first, *rest = value
                func = functools.singledispatchmethod(first)
                for impl in rest:
                    func.register(impl)
                namespace[key] = func
            elif all(isinstance(v, classmethod) for v in value):
                first, *rest = value
                func = functools.singledispatchmethod(first.__func__)
                for v in rest:
                    func.register(v.__func__)
                namespace[key] = classmethod(func)
            elif all(isinstance(v, staticmethod) for v in value):
                first, *rest = value
                func = functools.singledispatch(first.__func__)
                for v in rest:
                    func.register(v.__func__)
                namespace[key] = staticmethod(func)
            else:
                raise TypeError(f"Multiple attributes are defined with name {key}")

        return type.__new__(cls, name, bases, namespace)

    @classmethod
    def __prepare__(cls, name, bases):
        return _MultiDict()


class Dispatched(metaclass=DispatchedMeta):
    """Base class supporting multiple implementations of a method.

    Methods with the same name can be defined multiple times. The first method
    defined is the default implementation, and subsequent methods are registered
    as implementations for specific types of the first argument.

    The constructed methods are equivalent as if they were defined with
    `functools.singledispatchmethod` but without the need to use the decorator
    syntax. The recommended application of this class is to implement visitor
    patterns.

    Besides ordinary methods, classmethods and staticmethods are also supported.
    The implementation can be extended to overload multiple arguments by using
    `multimethod` instead of `singledispatchmethod` as the dispatcher.
    """
