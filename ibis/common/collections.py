from __future__ import annotations

import collections.abc
from abc import abstractmethod
from itertools import tee
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from public import public

from ibis.common.bases import Abstract, Hashable

if TYPE_CHECKING:
    from typing_extensions import Self

K = TypeVar("K", bound=collections.abc.Hashable)
V = TypeVar("V")


# The following classes provide an alternative to the `collections.abc` module
# which can be used with `ibis.common.bases` without metaclass conflicts but
# remains compatible with the `collections.abc` module. The main advantage is
# faster `isinstance` checks.


@collections.abc.Iterable.register
class Iterable(Abstract, Generic[V]):
    """Iterable abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __iter__(self):
        ...


@collections.abc.Reversible.register
class Reversible(Iterable[V]):
    """Reverse iterable abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __reversed__(self):
        ...


@collections.abc.Iterator.register
class Iterator(Iterable[V]):
    """Iterator abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __next__(self):
        ...

    def __iter__(self):
        return self


@collections.abc.Sized.register
class Sized(Abstract):
    """Sized abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __len__(self):
        ...


@collections.abc.Container.register
class Container(Abstract, Generic[V]):
    """Container abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __contains__(self, x):
        ...


@collections.abc.Collection.register
class Collection(Sized, Iterable[V], Container[V]):
    """Collection abstract base class for quicker isinstance checks."""


@collections.abc.Sequence.register
class Sequence(Reversible[V], Collection[V]):
    """Sequence abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __getitem__(self, index):
        ...

    def __iter__(self):
        i = 0
        try:
            while True:
                yield self[i]
                i += 1
        except IndexError:
            return

    def __contains__(self, value):
        return any(v is value or v == value for v in self)

    def __reversed__(self):
        for i in reversed(range(len(self))):
            yield self[i]

    def index(self, value, start=0, stop=None):
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)

        i = start
        while stop is None or i < stop:
            try:
                v = self[i]
            except IndexError:
                break
            if v is value or v == value:
                return i
            i += 1
        raise ValueError

    def count(self, value):
        return sum(1 for v in self if v is value or v == value)


@collections.abc.Mapping.register
class Mapping(Collection[K], Generic[K, V]):
    """Mapping abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __getitem__(self, key):
        ...

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def keys(self):
        return collections.abc.KeysView(self)

    def items(self):
        return collections.abc.ItemsView(self)

    def values(self):
        return collections.abc.ValuesView(self)

    def __eq__(self, other):
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())


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
    ...     __slots__ = ("_data",)
    ...
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

    def _check_conflict(self, other: collections.abc.Mapping) -> set[K]:
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

    def __ge__(self, other: collections.abc.Mapping) -> bool:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        return other.keys() == common_keys

    def __gt__(self, other: collections.abc.Mapping) -> bool:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        return len(self) > len(other) and self.__ge__(other)

    def __le__(self, other: collections.abc.Mapping) -> bool:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        return self.keys() == common_keys

    def __lt__(self, other: collections.abc.Mapping) -> bool:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        return len(self) < len(other) and self.__le__(other)

    def __and__(self, other: collections.abc.Mapping) -> Self:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        intersection = {k: v for k, v in self.items() if k in common_keys}
        return self.__class__(intersection)

    def __sub__(self, other: collections.abc.Mapping) -> Self:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        difference = {k: v for k, v in self.items() if k not in common_keys}
        return self.__class__(difference)

    def __rsub__(self, other: collections.abc.Mapping) -> Self:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        common_keys = self._check_conflict(other)
        difference = {k: v for k, v in other.items() if k not in common_keys}
        return self.__class__(difference)

    def __or__(self, other: collections.abc.Mapping) -> Self:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        self._check_conflict(other)
        union = {**self, **other}
        return self.__class__(union)

    def __xor__(self, other: collections.abc.Mapping) -> Self:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        left = self - other
        right = other - self
        left._check_conflict(right)
        union = {**left, **right}
        return self.__class__(union)

    def isdisjoint(self, other: collections.abc.Mapping) -> bool:
        common_keys = self._check_conflict(other)
        return not common_keys


@public
class FrozenDict(Mapping[K, V], Hashable):
    """Immutable dictionary with a precomputed hash value."""

    __slots__ = ("__view__", "__precomputed_hash__")
    __view__: MappingProxyType
    __precomputed_hash__: int

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


class RewindableIterator(Iterator[V]):
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

    __slots__ = ("_iterator", "_checkpoint")

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


# Need to provide type hint as else a static type checker does not recognize
# that frozendict exists in this module
frozendict: type[FrozenDict]
public(frozendict=FrozenDict)
