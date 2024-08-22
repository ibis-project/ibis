from __future__ import annotations

import functools
import sys
import weakref
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    import ibis.expr.operations as ops
    from ibis.backends import BaseBackend


def memoize(func: Callable) -> Callable:
    """Memoize a function."""
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        try:
            return cache[key]
        except KeyError:
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    return wrapper


class CacheEntry(NamedTuple):
    name: str
    ref: weakref.ReferenceType[ops.DatabaseTable]
    finalizer: weakref.finalize


class QueryCache:
    """A cache for executed Ibis expressions.

    Typically implemented by storing computed expressions as temporary tables
    in the backend.

    Notes
    -----
    We could implement `MutableMapping`, but the `__setitem__` implementation
    doesn't make sense and the `len` and `__iter__` methods aren't used.

    We can implement that interface if and when we need to.
    """

    def __init__(self, backend: BaseBackend) -> None:
        self.backend = backend
        self.cache: MutableMapping[ops.Relation, CacheEntry] = {}

    def get(self, key, default=None):
        if (entry := self.cache.get(key)) is not None:
            op = entry.ref()
            return op if op is not None else default
        return default

    def __getitem__(self, key: ops.Relation) -> ops.DatabaseTable:
        op = self.cache[key].ref()
        if op is None:
            raise KeyError(key)
        return op

    def store(self, input):
        """Compute and store a reference to `key`."""
        from ibis.util import gen_name

        key = input.op()
        name = gen_name("cache")

        self.backend._load_into_cache(name, input)

        cached = self.backend.table(name).op()
        finalizer = weakref.finalize(cached, self._release, key)

        self.cache[key] = CacheEntry(name, weakref.ref(cached), finalizer)

        return cached

    def release(self, name: str) -> None:
        """Release a cached table by `name`."""
        # Could be sped up with an inverse dictionary
        for key, entry in self.cache.items():
            if entry.name == name:
                self._release(key)
                return

    def _release(self, key: ops.Relation) -> None:
        """Release a cached table by `key` relation."""
        entry = self.cache.pop(key)
        try:
            self.backend._clean_up_cached_table(entry.name)
        except Exception:
            # suppress exceptions during interpreter shutdown
            if not sys.is_finalizing():
                raise
        entry.finalizer.detach()
