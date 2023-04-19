from __future__ import annotations

from types import MappingProxyType
from typing import Any, Hashable, Iterable, Iterator, Mapping, Set, TypeVar

from public import public
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
        return f"{self.__class__.__name__}({self.__view__!r})"

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


class DisjointSet(Mapping[K, Set[K]]):
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

    def __eq__(self, other: Self[K]) -> bool:
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


public(frozendict=FrozenDict, dotdict=DotDict)
