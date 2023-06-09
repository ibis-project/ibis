from __future__ import annotations

from collections.abc import Iterator
from itertools import tee
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Hashable, Mapping, TypeVar

from public import public

if TYPE_CHECKING:
    from typing_extensions import Self

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@public
class MapSet(Mapping[K, V]):
    """A mapping that also supports set-like operations.

    It is an altered version of `collections.abc.Mapping` that supports set-like
    operations. The `__iter__`, `__len__`, and `__getitem__` methods must be
    implemented.

    The set-like operations' other operand must be a `Mapping`. If the two
    operands contain common keys but with different values, then the operation
    becomes ambiguous and an exception will be raised.

    Examples
    --------
    >>> from ibis.common.collections import MapSet
    >>> class MyMap(MapSet):
    ...     def __init__(self, *args, **kwargs):
    ...         self._data = dict(*args, **kwargs)
    ...
    ...     def __iter__(self):
    ...         return iter(self._data)
    ...
    ...     def __len__(self):
    ...         return len(self._data)
    ...
    ...     def __getitem__(self, key):
    ...         return self._data[key]
    ...
    ...     def __repr__(self):
    ...         return f"MyMap({repr(self._data)})"
    ...
    >>> m = MyMap(a=1, b=2)
    >>> n = dict(a=1, b=2, c=3)
    >>> m <= n
    True
    >>> m < n
    True
    >>> n - m
    MyMap({'c': 3})
    >>> m & n
    MyMap({'a': 1, 'b': 2})
    >>> m | n
    MyMap({'a': 1, 'b': 2, 'c': 3})
    """

    def _check_conflict(self, other: Mapping) -> set[K]:
        # Check if there are conflicting key-value pairs between self and other.
        # A key-value pair is conflicting if the key is the same but the value is
        # different.
        common_keys = self.keys() & other.keys()
        for key in common_keys:
            left, right = self[key], other[key]
            if left != right:
                raise ValueError(
                    f"Conflicting values for key `{key}`: {left} != {right}"
                )
        return common_keys

    def __ge__(self, other: Mapping) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        return other.keys() == common_keys

    def __gt__(self, other: Mapping) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return len(self) > len(other) and self.__ge__(other)

    def __le__(self, other: Mapping) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        return self.keys() == common_keys

    def __lt__(self, other: Mapping) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return len(self) < len(other) and self.__le__(other)

    def __and__(self, other: Mapping) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        intersection = {k: v for k, v in self.items() if k in common_keys}
        return self.__class__(intersection)

    def __sub__(self, other: Mapping) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        difference = {k: v for k, v in self.items() if k not in common_keys}
        return self.__class__(difference)

    def __rsub__(self, other: Self) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        difference = {k: v for k, v in other.items() if k not in common_keys}
        return self.__class__(difference)

    def __or__(self, other: Mapping) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        self._check_conflict(other)
        union = {**self, **other}
        return self.__class__(union)

    def __xor__(self, other: Mapping) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        left = self - other
        right = other - self
        left._check_conflict(right)
        union = {**left, **right}
        return self.__class__(union)

    def isdisjoint(self, other: Mapping) -> bool:
        common_keys = self._check_conflict(other)
        return not common_keys


@public
class FrozenDict(Mapping[K, V], Hashable):
    """Immutable dictionary with a precomputed hash value."""

    __slots__ = ("__view__", "__precomputed_hash__")

    def __init__(self, *args, **kwargs):
        dictview = MappingProxyType(dict(*args, **kwargs))
        dicthash = hash(tuple(dictview.items()))
        object.__setattr__(self, "__view__", dictview)
        object.__setattr__(self, "__precomputed_hash__", dicthash)

    def __str__(self):
        return str(self.__view__)

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self.__view__)!r})"

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(f"Attribute {name!r} cannot be assigned to frozendict")

    def __reduce__(self):
        return self.__class__, (dict(self.__view__),)

    def __iter__(self):
        return iter(self.__view__)

    def __len__(self):
        return len(self.__view__)

    def __getitem__(self, key):
        return self.__view__[key]

    def __hash__(self):
        return self.__precomputed_hash__


@public
class DotDict(dict):
    """Dictionary that allows access to keys as attributes using the dot notation.

    Note, that this is not recursive, so nested access is not supported.

    Examples
    --------
    >>> d = DotDict({'a': 1, 'b': 2})
    >>> d.a
    1
    >>> d.b
    2
    >>> d['a']
    1
    >>> d['b']
    2
    >>> d.c = 3
    >>> d['c']
    3
    >>> d.c
    3
    """

    __slots__ = ()

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


class RewindableIterator(Iterator):
    """Iterator that can be rewound to a checkpoint.

    Examples
    --------
    >>> it = RewindableIterator(range(5))
    >>> next(it)
    0
    >>> next(it)
    1
    >>> it.checkpoint()
    >>> next(it)
    2
    >>> next(it)
    3
    >>> it.rewind()
    >>> next(it)
    2
    >>> next(it)
    3
    >>> next(it)
    4
    """

    def __init__(self, iterable):
        self._iterator = iter(iterable)
        self._checkpoint = None

    def __next__(self):
        return next(self._iterator)

    def rewind(self):
        """Rewind the iterator to the last checkpoint."""
        if self._checkpoint is None:
            raise ValueError("No checkpoint to rewind to.")
        self._iterator, self._checkpoint = tee(self._checkpoint)

    def checkpoint(self):
        """Create a checkpoint of the current iterator state."""
        self._iterator, self._checkpoint = tee(self._iterator)


public(frozendict=FrozenDict, dotdict=DotDict)
