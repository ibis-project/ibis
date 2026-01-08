from __future__ import annotations

import abc
import functools
from collections import defaultdict
from typing import Any, Callable, Generic, TypeVar, Union, overload

from ibis.common.typing import UnionType, evaluate_annotations, get_args, get_origin
from ibis.util import import_object, unalias_package

IntoType = Union[tuple, type, abc.ABCMeta, str]
R = TypeVar("R")


class _SingleDispatch:
    def __init__(self, func, typ=None):
        self.lookup = {}
        self.abc_lookup = {}
        self.lazy_lookup = defaultdict(dict)
        self.func = func
        self.add(func, typ)

    def add(self, func, typ: IntoType | None = None):
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

    @overload
    def register(
        self, typ: IntoType | None, func: None
    ) -> Callable[[Callable[..., R]], Callable[..., R]]: ...
    @overload
    def register(
        self, typ: IntoType | None, func: Callable[..., R]
    ) -> Callable[..., R]: ...

    def register(self, typ: IntoType | None, func=None):
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


class SingleDispatch(Generic[R]):
    def __init__(self, func: Callable[..., R]) -> None:
        self._dispatcher = _SingleDispatch(func, object)
        self._finalized = False
        self.__module__ = func.__module__
        self.__doc__ = func.__doc__
        self.__name__ = getattr(func, "__name__", None)
        self.__qualname__ = getattr(func, "__qualname__", None)
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__type_params__ = getattr(func, "__type_params__", ())

    def __call__(self, arg: Any, *args: Any, **kwargs: Any) -> R:
        impl = self._dispatcher.dispatch(type(arg))
        return impl(arg, *args, **kwargs)

    def dispatch(self, typ: type) -> Callable[..., R]:
        """Return the implementation for the given type."""
        return self._dispatcher.dispatch(typ)

    @overload
    def register(
        self, typ: IntoType, func: None = None
    ) -> Callable[[Callable[..., R]], Callable[..., R]]: ...
    @overload
    def register(self, typ: IntoType, func: Callable[..., R]) -> Callable[..., R]: ...

    def register(self, typ, func=None):
        """Register a new implementation for arguments of type `cls`."""
        if self._finalized:
            raise RuntimeError(
                "Cannot register new implementations on finalized dispatcher"
            )
        return self._dispatcher.register(typ, func)

    def finalize(self) -> None:
        """Prevent further registrations."""
        self._finalized = True


def lazy_singledispatch(func: Callable[..., R]) -> SingleDispatch[R]:
    """A `singledispatch` implementation that supports lazily registering implementations."""
    return SingleDispatch(func)
