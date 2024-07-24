from __future__ import annotations

import functools
import sys
from collections import namedtuple
from typing import TYPE_CHECKING, Any
from weakref import finalize, ref

if TYPE_CHECKING:
    from collections.abc import Callable


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


CacheEntry = namedtuple("CacheEntry", ["name", "ref", "finalizer"])


class RefCountedCache:
    """A cache with implicitly reference-counted values.

    We could implement `MutableMapping`, but the `__setitem__` implementation
    doesn't make sense and the `len` and `__iter__` methods aren't used.

    We can implement that interface if and when we need to.
    """

    def __init__(
        self,
        *,
        populate: Callable[[str, Any], None],
        lookup: Callable[[str], Any],
        finalize: Callable[[Any], None],
    ) -> None:
        self.populate = populate
        self.lookup = lookup
        self.finalize = finalize

        self.cache: dict[Any, CacheEntry] = dict()

    def get(self, key, default=None):
        if (entry := self.cache.get(key)) is not None:
            op = entry.ref()
            return op if op is not None else default
        return default

    def __getitem__(self, key):
        op = self.cache[key].ref()
        if op is None:
            raise KeyError(key)
        return op

    def store(self, input):
        """Compute and store a reference to `key`."""
        from ibis.util import gen_name

        key = input.op()
        name = gen_name("cache")
        self.populate(name, input)
        cached = self.lookup(name)
        finalizer = finalize(cached, self._release, key)

        self.cache[key] = CacheEntry(name, ref(cached), finalizer)

        return cached

    def release(self, name: str) -> None:
        # Could be sped up with an inverse dictionary
        for key, entry in self.cache.items():
            if entry.name == name:
                self._release(key)
                return

    def _release(self, key) -> None:
        entry = self.cache.pop(key)
        try:
            self.finalize(entry.name)
        except Exception:
            # suppress exceptions during interpreter shutdown
            if not sys.is_finalizing():
                raise
        entry.finalizer.detach()
