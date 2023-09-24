from __future__ import annotations

import collections.abc

import pytest

from ibis.common.collections import (
    Collection,
    Container,
    FrozenDict,
    Iterable,
    Iterator,
    Mapping,
    MapSet,
    Reversible,
    RewindableIterator,
    Sequence,
    Sized,
)
from ibis.tests.util import assert_pickle_roundtrip


def test_iterable():
    class MyIterable(Iterable):
        __slots__ = ("n",)

        def __init__(self, n):
            self.n = n

        def __iter__(self):
            return iter(range(self.n))

    i = MyIterable(10)
    assert isinstance(i, Iterable)
    assert isinstance(i, collections.abc.Iterable)
    assert list(i) == list(range(10))


def test_reversible():
    class MyReversible(Reversible):
        __slots__ = ("n",)

        def __init__(self, n):
            self.n = n

        def __iter__(self):
            return iter(range(self.n))

        def __reversed__(self):
            return reversed(range(self.n))

    r = MyReversible(10)
    assert isinstance(r, Reversible)
    assert isinstance(r, Iterable)
    assert isinstance(r, collections.abc.Reversible)
    assert isinstance(r, collections.abc.Iterable)
    assert list(r) == list(range(10))
    assert list(reversed(r)) == list(reversed(range(10)))


def test_iterator():
    class MyIterator(Iterator):
        __slots__ = ("n", "i")

        def __init__(self, n):
            self.n = n
            self.i = iter(range(n))

        def __next__(self):
            return next(self.i)

    i = MyIterator(10)
    assert isinstance(i, Iterator)
    assert isinstance(i, Iterable)
    assert isinstance(i, collections.abc.Iterator)
    assert isinstance(i, collections.abc.Iterable)
    for j in range(10):
        assert next(i) == j


def test_sized():
    class MySized(Sized):
        __slots__ = ("n",)

        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    s = MySized(10)
    assert isinstance(s, Sized)
    assert isinstance(s, collections.abc.Sized)
    assert len(s) == 10


def test_container():
    class MyContainer(Container):
        __slots__ = ("n",)

        def __init__(self, n):
            self.n = n

        def __contains__(self, x):
            return x in range(self.n)

    c = MyContainer(10)
    assert isinstance(c, Container)
    assert isinstance(c, collections.abc.Container)
    assert 5 in c
    assert 10 not in c


def test_collection():
    class MyCollection(Collection):
        __slots__ = ("n",)

        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(range(self.n))

        def __contains__(self, x):
            return x in range(self.n)

    c = MyCollection(10)
    assert isinstance(c, Collection)
    assert isinstance(c, Sized)
    assert isinstance(c, Iterable)
    assert isinstance(c, Container)
    assert isinstance(c, collections.abc.Collection)
    assert isinstance(c, collections.abc.Sized)
    assert isinstance(c, collections.abc.Iterable)
    assert isinstance(c, collections.abc.Container)
    assert len(c) == 10
    assert list(c) == list(range(10))
    assert 5 in c
    assert 10 not in c


def test_sequence():
    class MySequence(Sequence):
        __slots__ = ("n",)

        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, index):
            return list(range(self.n))[index]

    s = MySequence(10)
    assert isinstance(s, Sequence)
    assert isinstance(s, Reversible)
    assert isinstance(s, Collection)
    assert isinstance(s, Sized)
    assert isinstance(s, Iterable)
    assert isinstance(s, Container)
    assert isinstance(s, collections.abc.Sequence)
    assert isinstance(s, collections.abc.Reversible)
    assert isinstance(s, collections.abc.Collection)
    assert isinstance(s, collections.abc.Sized)
    assert isinstance(s, collections.abc.Iterable)
    assert isinstance(s, collections.abc.Container)
    assert len(s) == 10
    assert list(s) == list(range(10))
    assert 5 in s
    assert 10 not in s
    assert s[5] == 5
    assert s[-1] == 9
    assert s[1:5] == [1, 2, 3, 4]


def test_mapping():
    class MyMapping(Mapping):
        __slots__ = ("wrapped",)

        def __init__(self, **kwargs):
            self.wrapped = dict(kwargs)

        def __getitem__(self, key):
            return self.wrapped[key]

        def __len__(self):
            return len(self.wrapped)

        def __iter__(self):
            return iter(self.wrapped)

    m = MyMapping(a=1, b=2, c=3)
    assert isinstance(m, Mapping)
    assert isinstance(m, Collection)
    assert isinstance(m, Sized)
    assert isinstance(m, Iterable)
    assert isinstance(m, Container)
    assert isinstance(m, collections.abc.Mapping)
    assert isinstance(m, collections.abc.Collection)
    assert isinstance(m, collections.abc.Sized)
    assert isinstance(m, collections.abc.Iterable)
    assert isinstance(m, collections.abc.Container)
    assert len(m) == 3
    assert list(m) == ["a", "b", "c"]
    assert "a" in m
    assert "d" not in m
    assert m["a"] == 1
    assert m["b"] == 2
    assert m["c"] == 3
    assert m.get("a") == 1
    assert m.get("d") is None
    assert m.get("d", 4) == 4
    assert m == m
    assert m != MyMapping(a=1, b=2, c=3, d=4)
    assert list(m.keys()) == ["a", "b", "c"]
    assert list(m.values()) == [1, 2, 3]
    assert list(m.items()) == [("a", 1), ("b", 2), ("c", 3)]
    assert isinstance(m.keys(), collections.abc.KeysView)
    assert isinstance(m.values(), collections.abc.ValuesView)
    assert isinstance(m.items(), collections.abc.ItemsView)
    assert m == dict(a=1, b=2, c=3)
    assert m != dict(a=1, b=2, c=3, d=4)


class MySchema(MapSet):
    __slots__ = ("_fields",)

    def __init__(self, dct=None, **kwargs):
        self._fields = dict(dct or kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._fields})"

    def __getitem__(self, key):
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def identical(self, other):
        return type(self) == type(other) and tuple(self.items()) == tuple(other.items())


def test_myschema_identical():
    ms1 = MySchema(a=1, b=2)
    ms2 = MySchema(a=1, b=2)
    ms3 = MySchema(b=2, a=1)
    ms4 = MySchema(a=1, b=2, c=3)
    ms5 = {}

    assert ms1.identical(ms2)
    assert not ms1.identical(ms3)
    assert not ms1.identical(ms4)
    assert not ms1.identical(ms5)


def test_mapset_mapping_api():
    ms = MySchema(a=1, b=2)

    assert isinstance(ms, MapSet)
    assert isinstance(ms, Mapping)
    assert isinstance(ms, collections.abc.Mapping)
    assert ms["a"] == 1
    assert ms["b"] == 2
    assert len(ms) == 2
    assert isinstance(iter(ms), collections.abc.Iterator)
    assert list(ms) == ["a", "b"]
    assert isinstance(ms.keys(), collections.abc.KeysView)
    assert list(ms.keys()) == ["a", "b"]
    assert isinstance(ms.values(), collections.abc.ValuesView)
    assert list(ms.values()) == [1, 2]
    assert isinstance(ms.items(), collections.abc.ItemsView)
    assert list(ms.items()) == [("a", 1), ("b", 2)]
    assert ms.get("a") == 1
    assert ms.get("c") is None
    assert ms.get("c", 3) == 3
    assert "a" in ms
    assert "c" not in ms
    assert ms == ms
    assert ms != MySchema(a=1, b=2, c=3)


def test_mapset_set_api():
    a = MySchema(a=1, b=2)
    a_ = MySchema(a=1, b=-2)
    b = MySchema(a=1, b=2, c=3)
    b_ = MySchema(a=1, b=2, c=-3)
    f = MySchema(d=4, e=5)

    # disjoint
    assert not a.isdisjoint(b)
    assert a.isdisjoint(f)

    # __eq__, __ne__
    assert a == a
    assert a != a_
    assert b == b
    assert b != b_

    # __le__, __lt__
    assert a < b
    assert a < dict(b)
    assert a <= b
    assert a <= dict(b)
    assert a <= a
    assert a <= dict(a)
    assert not b <= a
    assert not b <= dict(a)
    assert not b < a
    assert not b < dict(a)
    with pytest.raises(ValueError, match="Conflicting values"):
        # duplicate keys with different values
        a <= a_  # noqa: B015
    with pytest.raises(ValueError, match="Conflicting values"):
        a <= dict(a_)  # noqa: B015
    with pytest.raises(TypeError, match="not supported"):
        a < 1  # noqa: B015
    with pytest.raises(TypeError, match="not supported"):
        a <= 1  # noqa: B015

    # __gt__, __ge__
    assert b > a
    assert b > dict(a)
    assert b >= a
    assert b >= dict(a)
    assert a >= a
    assert a >= dict(a)
    assert not a >= b
    assert not a >= dict(b)
    assert not a > b
    assert not a > dict(b)
    assert not a_ > a
    assert not a_ > dict(a)
    with pytest.raises(ValueError, match="Conflicting values"):
        a_ >= a  # noqa: B015
    with pytest.raises(ValueError, match="Conflicting values"):
        a_ >= dict(a)  # noqa: B015
    with pytest.raises(TypeError, match="not supported"):
        a > 1  # noqa: B015
    with pytest.raises(TypeError, match="not supported"):
        a >= 1  # noqa: B015

    # __and__
    with pytest.raises(ValueError, match="Conflicting values"):
        a & a_
    with pytest.raises(ValueError, match="Conflicting values"):
        a & dict(a_)
    with pytest.raises(ValueError, match="Conflicting values"):
        b & b_
    with pytest.raises(ValueError, match="Conflicting values"):
        b & dict(b_)
    assert (a & b).identical(a)
    assert (a & dict(b)).identical(a)
    assert (a & f).identical(MySchema())
    assert (a & dict(f)).identical(MySchema())
    with pytest.raises(TypeError, match="unsupported operand"):
        a & 1

    # __or__
    assert (a | a).identical(a)
    assert (a | dict(a)).identical(a)
    assert (a | b).identical(b)
    assert (a | f).identical(MySchema(a=1, b=2, d=4, e=5))
    with pytest.raises(ValueError, match="Conflicting values"):
        a | a_
    with pytest.raises(ValueError, match="Conflicting values"):
        a | dict(a_)
    with pytest.raises(TypeError, match="unsupported operand"):
        a | 1

    # __sub__
    with pytest.raises(ValueError, match="Conflicting values"):
        a - a_
    with pytest.raises(ValueError, match="Conflicting values"):
        a - dict(a_)
    assert (a - b).identical(MySchema())
    assert (a - dict(b)).identical(MySchema())
    assert (b - a).identical(MySchema(c=3))
    assert (dict(b) - a).identical(MySchema(c=3))
    assert (a - f).identical(a)
    assert (a - dict(f)).identical(a)
    assert (f - a).identical(f)
    with pytest.raises(TypeError, match="unsupported operand"):
        f - 1

    # __xor__
    with pytest.raises(ValueError, match="Conflicting values"):
        a ^ a_
    with pytest.raises(ValueError, match="Conflicting values"):
        a ^ dict(a_)

    assert (a ^ b).identical(MySchema(c=3))
    assert (a ^ dict(b)).identical(MySchema(c=3))
    assert (b ^ a).identical(MySchema(c=3))
    assert (a ^ f).identical(MySchema(a=1, b=2, d=4, e=5))
    assert (f ^ a).identical(MySchema(d=4, e=5, a=1, b=2))


def test_frozendict():
    d = FrozenDict({"a": 1, "b": 2, "c": 3})
    e = FrozenDict(a=1, b=2, c=3)
    f = FrozenDict(a=1, b=2, c=3, d=4)

    assert isinstance(d, Mapping)
    assert isinstance(d, collections.abc.Mapping)

    assert d == e
    assert d != f

    assert d["a"] == 1
    assert d["b"] == 2

    msg = "'FrozenDict' object does not support item assignment"
    with pytest.raises(TypeError, match=msg):
        d["a"] = 2
    with pytest.raises(TypeError, match=msg):
        d["d"] = 4

    with pytest.raises(TypeError):
        d.__view__["a"] = 2
    with pytest.raises(TypeError):
        d.__view__ = {"a": 2}

    assert hash(d)
    assert_pickle_roundtrip(d)


def test_rewindable_iterator():
    it = RewindableIterator(range(10))
    assert next(it) == 0
    assert next(it) == 1
    with pytest.raises(ValueError, match="No checkpoint to rewind to"):
        it.rewind()

    it.checkpoint()
    assert next(it) == 2
    assert next(it) == 3
    it.rewind()
    assert next(it) == 2
    assert next(it) == 3
    assert next(it) == 4
    it.checkpoint()
    assert next(it) == 5
    assert next(it) == 6
    it.rewind()
    assert next(it) == 5
    assert next(it) == 6
    assert next(it) == 7
    it.rewind()
    assert next(it) == 5
    assert next(it) == 6
    assert next(it) == 7
    assert next(it) == 8
    assert next(it) == 9
    with pytest.raises(StopIteration):
        next(it)
