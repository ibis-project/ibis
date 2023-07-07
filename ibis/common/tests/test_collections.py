from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView

import pytest

from ibis.common.collections import DotDict, FrozenDict, MapSet, RewindableIterator
from ibis.tests.util import assert_pickle_roundtrip


class MySchema(MapSet):
    def __init__(self, dct=None, **kwargs):
        self._fields = dict(dct or kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._fields})'

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
    assert ms['a'] == 1
    assert ms['b'] == 2
    assert len(ms) == 2
    assert isinstance(iter(ms), Iterator)
    assert list(ms) == ['a', 'b']
    assert isinstance(ms.keys(), KeysView)
    assert list(ms.keys()) == ['a', 'b']
    assert isinstance(ms.values(), ValuesView)
    assert list(ms.values()) == [1, 2]
    assert isinstance(ms.items(), ItemsView)
    assert list(ms.items()) == [('a', 1), ('b', 2)]
    assert ms.get('a') == 1
    assert ms.get('c') is None
    assert ms.get('c', 3) == 3
    assert 'a' in ms
    assert 'c' not in ms
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


def test_dotdict():
    d = DotDict({"a": 1, "b": 2, "c": 3})
    assert d["a"] == d.a == 1
    assert d["b"] == d.b == 2

    d.b = 3
    assert d.b == 3
    assert d["b"] == 3

    del d.c
    assert not hasattr(d, "c")
    assert "c" not in d

    assert repr(d) == "DotDict({'a': 1, 'b': 3})"

    with pytest.raises(KeyError):
        assert d['x']
    with pytest.raises(AttributeError):
        assert d.x


def test_frozendict():
    d = FrozenDict({"a": 1, "b": 2, "c": 3})
    e = FrozenDict(a=1, b=2, c=3)
    f = FrozenDict(a=1, b=2, c=3, d=4)
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
