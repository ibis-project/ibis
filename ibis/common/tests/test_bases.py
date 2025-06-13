from __future__ import annotations

import copy
import pickle
import weakref
from abc import ABCMeta, abstractmethod

import pytest

from ibis.common.bases import (
    Abstract,
    AbstractMeta,
    Comparable,
    Final,
    FrozenSlotted,
    Immutable,
    Singleton,
    Slotted,
)


def test_classes_are_based_on_abstract():
    assert issubclass(Comparable, Abstract)
    assert issubclass(Final, Abstract)
    assert issubclass(Immutable, Abstract)
    assert issubclass(Singleton, Abstract)


def test_abstract():
    class Foo(Abstract):
        @abstractmethod
        def foo(self): ...

        @property
        @abstractmethod
        def bar(self): ...

    assert not issubclass(type(Foo), ABCMeta)
    assert issubclass(type(Foo), AbstractMeta)
    assert Foo.__abstractmethods__ == frozenset({"foo", "bar"})

    with pytest.raises(TypeError, match="Can't instantiate abstract class .*Foo.*"):
        Foo()

    class Bar(Foo):
        def foo(self):
            return 1

        @property
        def bar(self):
            return 2

    bar = Bar()
    assert bar.foo() == 1
    assert bar.bar == 2
    assert isinstance(bar, Foo)
    assert isinstance(bar, Abstract)
    assert Bar.__abstractmethods__ == frozenset()


def test_immutable():
    class Foo(Immutable):
        __slots__ = ("a", "b")

        def __init__(self, a, b):
            object.__setattr__(self, "a", a)
            object.__setattr__(self, "b", b)

    foo = Foo(1, 2)
    assert foo.a == 1
    assert foo.b == 2
    with pytest.raises(AttributeError):
        foo.a = 2
    with pytest.raises(AttributeError):
        foo.b = 3

    assert copy.copy(foo) is foo
    assert copy.deepcopy(foo) is foo


class Cache(dict):
    def setpair(self, a, b, value):
        a, b = id(a), id(b)
        self.setdefault(a, {})[b] = value
        self.setdefault(b, {})[a] = value

    def getpair(self, a, b):
        return self.get(id(a), {}).get(id(b))


class Node(Comparable):
    # override the default cache object
    __cache__ = Cache()
    __slots__ = ("name",)
    num_equal_calls = 0

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Node(name={self.name})"

    def __equals__(self, other):
        Node.num_equal_calls += 1
        return self.name == other.name


@pytest.fixture
def cache():
    Node.num_equal_calls = 0
    cache = Node.__cache__
    yield cache
    assert not cache


def test_comparable_basic():
    a = Node(name="a")
    b = Node(name="a")
    c = Node(name="a")
    assert a == b
    assert a == c
    del a
    del b
    del c


def test_comparable_caching(cache):
    a = Node(name="a")
    b = Node(name="b")
    c = Node(name="c")
    d = Node(name="d")
    e = Node(name="e")

    cache.setpair(a, b, True)
    cache.setpair(a, c, False)
    cache.setpair(c, d, True)
    cache.setpair(b, d, False)
    expected = {
        id(a): {id(b): True, id(c): False},
        id(b): {id(a): True, id(d): False},
        id(c): {id(a): False, id(d): True},
        id(d): {id(c): True, id(b): False},
    }
    assert cache == expected

    assert a == b
    assert b == a
    assert a != c
    assert c != a
    assert c == d
    assert d == c
    assert b != d
    assert d != b
    assert Node.num_equal_calls == 0
    assert cache == expected

    # no cache hit
    assert cache.getpair(a, e) is None
    assert a != e
    assert cache.getpair(a, e) is False
    assert Node.num_equal_calls == 1
    expected = {
        id(a): {id(b): True, id(c): False, id(e): False},
        id(b): {id(a): True, id(d): False},
        id(c): {id(a): False, id(d): True},
        id(d): {id(c): True, id(b): False},
        id(e): {id(a): False},
    }
    assert cache == expected

    # run only once
    assert e != a
    assert Node.num_equal_calls == 1
    assert cache.getpair(a, e) is False
    assert cache == expected


def test_comparable_garbage_collection(cache):
    a = Node(name="a")
    b = Node(name="b")
    c = Node(name="c")
    d = Node(name="d")

    cache.setpair(a, b, True)
    cache.setpair(a, c, False)
    cache.setpair(c, d, True)
    cache.setpair(b, d, False)

    assert cache.getpair(a, c) is False
    assert cache.getpair(c, d) is True
    del c
    assert cache == {
        id(a): {id(b): True},
        id(b): {id(a): True, id(d): False},
        id(d): {id(b): False},
    }

    assert cache.getpair(a, b) is True
    assert cache.getpair(b, d) is False
    del b
    assert cache == {}

    assert a != d
    assert cache == {id(a): {id(d): False}, id(d): {id(a): False}}
    del a
    assert cache == {}


def test_comparable_cache_reuse(cache):
    nodes = [
        Node(name="a"),
        Node(name="b"),
        Node(name="c"),
        Node(name="d"),
        Node(name="e"),
    ]

    expected = 0
    for a, b in zip(nodes, nodes):
        a == a  # noqa: B015
        a == b  # noqa: B015
        b == a  # noqa: B015
        if a != b:
            expected += 1
        assert Node.num_equal_calls == expected

    assert len(cache) == expected

    # check that cache is evicted once nodes get collected
    del nodes
    assert len(cache) == 0

    a = Node(name="a")
    b = Node(name="a")
    assert a == b


class OneAndOnly(Singleton):
    __instances__ = weakref.WeakValueDictionary()


class DataType(Singleton):
    __slots__ = ("nullable",)
    __instances__ = weakref.WeakValueDictionary()

    def __init__(self, nullable=True):
        self.nullable = nullable


def test_singleton_basics():
    one = OneAndOnly()
    only = OneAndOnly()
    assert one is only

    assert len(OneAndOnly.__instances__) == 1
    key = (OneAndOnly, (), ())
    assert OneAndOnly.__instances__[key] is one


def test_singleton_lifetime() -> None:
    one = OneAndOnly()
    assert len(OneAndOnly.__instances__) == 1

    del one
    assert len(OneAndOnly.__instances__) == 0


def test_singleton_with_argument() -> None:
    dt1 = DataType(nullable=True)
    dt2 = DataType(nullable=False)
    dt3 = DataType(nullable=True)

    assert dt1 is dt3
    assert dt1 is not dt2
    assert len(DataType.__instances__) == 2

    del dt3
    assert len(DataType.__instances__) == 2
    del dt1
    assert len(DataType.__instances__) == 1
    del dt2
    assert len(DataType.__instances__) == 0


def test_final():
    class A(Final):
        pass

    with pytest.raises(TypeError, match="Cannot inherit from final class .*A.*"):

        class B(A):
            pass


class MyObj(Slotted):
    __slots__ = ("a", "b")


def test_slotted():
    obj = MyObj(a=1, b=2)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.__fields__ == ("a", "b")
    assert obj.__slots__ == ("a", "b")
    with pytest.raises(AttributeError):
        obj.c = 3

    obj2 = MyObj(a=1, b=2)
    assert obj == obj2
    assert obj is not obj2

    obj3 = MyObj(a=1, b=3)
    assert obj != obj3

    assert pickle.loads(pickle.dumps(obj)) == obj

    with pytest.raises(KeyError):
        MyObj(a=1)


class MyObj2(MyObj):
    __slots__ = ("c",)


def test_slotted_inheritance():
    obj = MyObj2(a=1, b=2, c=3)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.__fields__ == ("a", "b", "c")
    assert obj.__slots__ == ("c",)
    with pytest.raises(AttributeError):
        obj.d = 4

    obj2 = MyObj2(a=1, b=2, c=3)
    assert obj == obj2
    assert obj is not obj2

    obj3 = MyObj2(a=1, b=2, c=4)
    assert obj != obj3
    assert pickle.loads(pickle.dumps(obj)) == obj

    with pytest.raises(KeyError):
        MyObj2(a=1, b=2)


class MyFrozenObj(FrozenSlotted):
    __slots__ = ("a", "b")


class MyFrozenObj2(MyFrozenObj):
    __slots__ = ("c", "d")


def test_frozen_slotted():
    obj = MyFrozenObj(a=1, b=2)

    assert obj.a == 1
    assert obj.b == 2
    assert obj.__fields__ == ("a", "b")
    assert obj.__slots__ == ("a", "b")
    with pytest.raises(AttributeError):
        obj.b = 3
    with pytest.raises(AttributeError):
        obj.c = 3

    obj2 = MyFrozenObj(a=1, b=2)
    assert obj == obj2
    assert obj is not obj2
    assert hash(obj) == hash(obj2)

    restored = pickle.loads(pickle.dumps(obj))
    assert restored == obj
    assert hash(restored) == hash(obj)

    with pytest.raises(KeyError):
        MyFrozenObj(a=1)


def test_frozen_slotted_inheritance():
    obj3 = MyFrozenObj2(a=1, b=2, c=3, d=4)
    assert obj3.__slots__ == ("c", "d")
    assert obj3.__fields__ == ("a", "b", "c", "d")
    assert pickle.loads(pickle.dumps(obj3)) == obj3
