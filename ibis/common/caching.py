from __future__ import annotations

import functools
from collections import Counter, defaultdict
from typing import Any, Callable

from bidict import bidict

from ibis.common.exceptions import IbisError


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


class RefCountedCache:
    """A cache with reference-counted keys.

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
        generate_name: Callable[[], str],
        key: Callable[[Any], Any],
    ) -> None:
        self.cache = bidict()
        # Somehow mypy needs a type hint here
        self.refs: Counter = Counter()
        self.populate = populate
        self.lookup = lookup
        self.finalize = finalize
        # Somehow mypy needs a type hint here
        self.names: defaultdict = defaultdict(generate_name)
        self.key = key or (lambda x: x)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        result = self.cache[key]
        self.refs[key] += 1
        return result

    def store(self, input) -> None:
        """Compute and store a reference to `key`."""
        key = self.key(input)
        name = self.names[key]
        self.populate(name, input)
        self.cache[key] = self.lookup(name)
        # nothing outside of this instance has referenced this key yet, so the
        # refcount is zero
        #
        # in theory it's possible to call store -> delitem which would raise an
        # exception, but in practice this doesn't happen because the only call
        # to store is immediately followed by a call to getitem.
        self.refs[key] = 0

    def __delitem__(self, key) -> None:
        # we need to remove the expression representing the computed physical
        # table as well as the expression that was used to create that table
        #
        # bidict automatically handles this for us; without it we'd have to do
        # to the bookkeeping ourselves with two dicts
        if (inv_key := self.cache.inverse.get(key)) is None:
            raise IbisError(
                "Key has already been released. Did you call "
                "`.release()` twice on the same expression?"
            )

        self.refs[inv_key] -= 1
        assert self.refs[inv_key] >= 0, f"refcount is negative: {self.refs[inv_key]:d}"

        if not self.refs[inv_key]:
            del self.cache[inv_key], self.refs[inv_key]
            self.finalize(key)
