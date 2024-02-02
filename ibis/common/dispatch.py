from __future__ import annotations

import abc
import functools
import inspect
import re
from collections import defaultdict

from ibis.common.typing import (
    Union,
    UnionType,
    evaluate_annotations,
    get_args,
    get_origin,
)
from ibis.util import import_object, unalias_package


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


class SingleDispatch:
    def __init__(self, func, typ=None):
        self.lookup = {}
        self.abc_lookup = {}
        self.lazy_lookup = defaultdict(dict)
        self.func = func
        self.add(func, typ)

    def add(self, func, typ=None):
        if typ is None:
            annots = getattr(func, "__annotations__", {})
            typehints = evaluate_annotations(annots, func.__module__, best_effort=True)
            if typehints:
                typ, *_ = typehints.values()
                if get_origin(typ) in (Union, UnionType):
                    for t in get_args(typ):
                        self.add(func, t)
                else:
                    self.add(func, typ)
            else:
                self.add(func, object)
        elif isinstance(typ, tuple):
            for t in typ:
                self.add(func, t)
        elif isinstance(typ, abc.ABCMeta):
            if typ in self.abc_lookup:
                raise TypeError(f"{typ} is already registered")
            self.abc_lookup[typ] = func
        elif isinstance(typ, str):
            package, rest = typ.split(".", 1)
            package = unalias_package(package)
            typ = f"{package}.{rest}"
            if typ in self.lazy_lookup[package]:
                raise TypeError(f"{typ} is already registered")
            self.lazy_lookup[package][typ] = func
        else:
            if typ in self.lookup:
                raise TypeError(f"{typ} is already registered")
            self.lookup[typ] = func
        return func

    def register(self, typ, func=None):
        """Register a new implementation for arguments of type `cls`."""

        def inner(func):
            self.add(func, typ)
            return func

        return inner if func is None else inner(func)

    def dispatch(self, typ):
        """Return the implementation for the given `cls`."""
        for klass in typ.__mro__:
            # 1. Check for a concrete implementation
            try:
                impl = self.lookup[klass]
            except KeyError:
                pass
            else:
                if typ is not klass:
                    # Cache implementation
                    self.lookup[typ] = impl
                return impl
            # 2. Check lazy implementations
            package = klass.__module__.split(".", 1)[0]
            if lazy := self.lazy_lookup.get(package):
                # Import all lazy implementations first before registering
                # (which should never fail), to ensure an error anywhere
                # doesn't result in a half-registered state.
                new = {import_object(name): func for name, func in lazy.items()}
                self.lookup.update(new)
                # drop lazy implementations, idempotent for thread safety
                self.lazy_lookup.pop(package, None)
                return self.dispatch(typ)
            # 3. Check for abcs
            for abc_class, impl in self.abc_lookup.items():
                if issubclass(typ, abc_class):
                    self.lookup[typ] = impl
                    return impl
        raise TypeError(f"Could not find implementation for {typ}")

    def __call__(self, arg, *args, **kwargs):
        impl = self.dispatch(type(arg))
        return impl(arg, *args, **kwargs)

    def __get__(self, obj, cls=None):
        def _method(*args, **kwargs):
            method = self.dispatch(type(args[0]))
            method = method.__get__(obj, cls)
            return method(*args, **kwargs)

        functools.update_wrapper(_method, self.func)
        return _method


def lazy_singledispatch(func):
    """A `singledispatch` implementation that supports lazily registering implementations."""

    dispatcher = SingleDispatch(func, object)

    @functools.wraps(func)
    def call(arg, *args, **kwargs):
        impl = dispatcher.dispatch(type(arg))
        return impl(arg, *args, **kwargs)

    call.dispatch = dispatcher.dispatch
    call.register = dispatcher.register
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
                func = SingleDispatch(first)
                for impl in rest:
                    func.add(impl)
                namespace[key] = func
            elif all(isinstance(v, classmethod) for v in value):
                first, *rest = value
                func = SingleDispatch(first.__func__)
                for impl in rest:
                    func.add(impl.__func__)
                namespace[key] = classmethod(func)
            elif all(isinstance(v, staticmethod) for v in value):
                first, *rest = value
                func = SingleDispatch(first.__func__)
                for impl in rest:
                    func.add(impl.__func__)
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
