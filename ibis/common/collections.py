from __future__ import annotations

import collections.abc
from abc import abstractmethod
from itertools import tee
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from koerce import AbstractMeta
from public import public

from ibis.common.exceptions import ConflictingValuesError

if TYPE_CHECKING:
    from typing_extensions import Self

K = TypeVar("K", bound=collections.abc.Hashable)
V = TypeVar("V")


@collections.abc.Iterable.register
class Iterable(Generic[V], metaclass=AbstractMeta):
    """Iterable abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __iter__(self): ...


@collections.abc.Reversible.register
class Reversible(Iterable[V]):
    """Reverse iterable abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __reversed__(self): ...


@collections.abc.Iterator.register
class Iterator(Iterable[V]):
    """Iterator abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __next__(self): ...

    def __iter__(self):
        return self


@collections.abc.Sized.register
class Sized(metaclass=AbstractMeta):
    """Sized abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __len__(self): ...


@collections.abc.Container.register
class Container(Generic[V], metaclass=AbstractMeta):
    """Container abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __contains__(self, x): ...


@collections.abc.Collection.register
class Collection(Sized, Iterable[V], Container[V]):
    """Collection abstract base class for quicker isinstance checks."""


@collections.abc.Sequence.register
class Sequence(Reversible[V], Collection[V]):
    """Sequence abstract base class for quicker isinstance checks."""

    @abstractmethod
    def __getitem__(self, index): ...

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
    def __getitem__(self, key): ...

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
class FrozenDict(dict[K, V], Mapping[K, V]):
    __slots__ = ("__precomputed_hash__",)
    # TODO(kszucs): Annotable is the base class, so traditional typehint is not allowed
    # __precomputed_hash__: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hashable = frozenset(self.items())
        object.__setattr__(self, "__precomputed_hash__", hash(hashable))

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __setitem__(self, key: K, value: V) -> None:
        raise TypeError(
            f"'{self.__class__.__name__}' object does not support item assignment"
        )

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(f"Attribute {name!r} cannot be assigned to frozendict")

    def __reduce__(self) -> tuple:
        return (self.__class__, (dict(self),))


@public
class FrozenOrderedDict(FrozenDict[K, V]):
    def __init__(self, *args, **kwargs):
        super(FrozenDict, self).__init__(*args, **kwargs)
        hashable = tuple(self.items())
        object.__setattr__(self, "__precomputed_hash__", hash(hashable))

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        return tuple(self.items()) == tuple(other.items())

    def __ne__(self, other: Any) -> bool:
        return not self == other


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
        conflicts = {
            (key, self[key], other[key])
            for key in common_keys
            if self[key] != other[key]
        }
        if conflicts:
            raise ConflictingValuesError(conflicts)
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


class DisjointSet(Mapping[K, set[K]]):
    """Disjoint set data structure.

    Also known as union-find data structure. It is a data structure that keeps
    track of a set of elements partitioned into a number of disjoint (non-overlapping)
    subsets. It provides near-constant-time operations to add new sets, to merge
    existing sets, and to determine whether elements are in the same set.

    Parameters
    ----------
    data :
        Initial data to add to the disjoint set.

    Examples
    --------
    >>> ds = DisjointSet()
    >>> ds.add(1)
    1
    >>> ds.add(2)
    2
    >>> ds.add(3)
    3
    >>> ds.union(1, 2)
    True
    >>> ds.union(2, 3)
    True
    >>> ds.find(1)
    1
    >>> ds.find(2)
    1
    >>> ds.find(3)
    1
    >>> ds.union(1, 3)
    False

    """

    __slots__ = ("_parents", "_classes")
    _parents: dict
    _classes: dict

    def __init__(self, data: Iterable[K] | None = None):
        self._parents = {}
        self._classes = {}
        if data is not None:
            for id in data:
                self.add(id)

    def __contains__(self, id) -> bool:
        """Check if the given id is in the disjoint set.

        Parameters
        ----------
        id :
            The id to check.

        Returns
        -------
        ined:
            True if the id is in the disjoint set, False otherwise.

        """
        return id in self._parents

    def __getitem__(self, id) -> set[K]:
        """Get the set of ids that are in the same class as the given id.

        Parameters
        ----------
        id :
            The id to get the class for.

        Returns
        -------
        class:
            The set of ids that are in the same class as the given id, including
            the given id.

        """
        id = self._parents[id]
        return self._classes[id]

    def __iter__(self) -> Iterator[K]:
        """Iterate over the ids in the disjoint set."""
        return iter(self._parents)

    def __len__(self) -> int:
        """Get the number of ids in the disjoint set."""
        return len(self._parents)

    def __eq__(self, other: object) -> bool:
        """Check if the disjoint set is equal to another disjoint set.

        Parameters
        ----------
        other :
            The other disjoint set to compare to.

        Returns
        -------
        equal:
            True if the disjoint sets are equal, False otherwise.

        """
        if not isinstance(other, DisjointSet):
            return NotImplemented
        return self._parents == other._parents

    def copy(self) -> DisjointSet:
        """Make a copy of the disjoint set.

        Returns
        -------
        copy:
            A copy of the disjoint set.

        """
        ds = DisjointSet()
        ds._parents = self._parents.copy()
        ds._classes = self._classes.copy()
        return ds

    def add(self, id: K) -> K:
        """Add a new id to the disjoint set.

        If the id is not in the disjoint set, it will be added to the disjoint set
        along with a new class containing only the given id.

        Parameters
        ----------
        id :
            The id to add to the disjoint set.

        Returns
        -------
        id:
            The id that was added to the disjoint set.

        """
        if id in self._parents:
            return self._parents[id]
        self._parents[id] = id
        self._classes[id] = {id}
        return id

    def find(self, id: K) -> K:
        """Find the root of the class that the given id is in.

        Also called as the canonicalized id or the representative id.

        Parameters
        ----------
        id :
            The id to find the canonicalized id for.

        Returns
        -------
        id:
            The canonicalized id for the given id.

        """
        return self._parents[id]

    def union(self, id1, id2) -> bool:
        """Merge the classes that the given ids are in.

        If the ids are already in the same class, this will return False. Otherwise
        it will merge the classes and return True.

        Parameters
        ----------
        id1 :
            The first id to merge the classes for.
        id2 :
            The second id to merge the classes for.

        Returns
        -------
        merged:
            True if the classes were merged, False otherwise.

        """
        # Find the root of each class
        id1 = self._parents[id1]
        id2 = self._parents[id2]
        if id1 == id2:
            return False

        # Merge the smaller eclass into the larger one, aka. union-find by size
        class1 = self._classes[id1]
        class2 = self._classes[id2]
        if len(class1) >= len(class2):
            id1, id2 = id2, id1
            class1, class2 = class2, class1

        # Update the parent pointers, this is called path compression but done
        # during the union operation to keep the find operation minimal
        for id in class1:
            self._parents[id] = id2

        # Do the actual merging and clear the other eclass
        class2 |= class1
        class1.clear()

        return True

    def connected(self, id1, id2):
        """Check if the given ids are in the same class.

        True if both ids have the same canonicalized id, False otherwise.

        Parameters
        ----------
        id1 :
            The first id to check.
        id2 :
            The second id to check.

        Returns
        -------
        connected:
            True if the ids are connected, False otherwise.

        """
        return self._parents[id1] == self._parents[id2]

    def verify(self):
        """Verify that the disjoint set is not corrupted.

        Check that each id's canonicalized id's class. In general corruption
        should not happen if the public API is used, but this is a sanity check
        to make sure that the internal data structures are not corrupted.

        Returns
        -------
        verified:
            True if the disjoint set is not corrupted, False otherwise.

        """
        for id in self._parents:
            if id not in self._classes[self._parents[id]]:
                raise RuntimeError(
                    f"DisjointSet is corrupted: {id} is not in its class"
                )


# Need to provide type hint as else a static type checker does not recognize
# that frozendict exists in this module
frozendict: type[FrozenDict]
public(frozendict=FrozenDict)
