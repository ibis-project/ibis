from __future__ import annotations

import weakref
from typing import MutableMapping


class WeakCache(MutableMapping):
    __slots__ = ('_data',)

    def __init__(self):
        object.__setattr__(self, '_data', {})

    def __setattr__(self, name, value):
        raise TypeError(f"can't set {name}")

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __setitem__(self, key, value):
        # construct an alternative representation of the key using the id()
        # of the key's components, this prevents infinite recursions
        identifiers = tuple(id(item) for item in key)

        # create a function which removes the key from the cache
        def callback(ref_):
            return self._data.pop(identifiers, None)

        # create weak references for the key's components with the callback
        # to remove the cache entry if any of the key's components gets
        # garbage collected
        refs = tuple(weakref.ref(item, callback) for item in key)

        self._data[identifiers] = (value, refs)

    def __getitem__(self, key):
        identifiers = tuple(id(item) for item in key)
        value, _ = self._data[identifiers]
        return value

    def __delitem__(self, key):
        identifiers = tuple(id(item) for item in key)
        del self._data[identifiers]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"
