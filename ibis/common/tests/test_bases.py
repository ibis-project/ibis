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
from ibis.common.caching import WeakCache


def test_classes_are_based_on_abstract():
    assert issubclass(Comparable, Abstract)
    assert issubclass(Final, Abstract)
    assert issubclass(Immutable, Abstract)
    assert issubclass(Singleton, Abstract)


def test_abstract():
    class Foo(Abstract):
        @abstractmethod
        def foo(self):
            ...

        @property
        @abstractmethod
        def bar(self):
            ...

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


class Node(Comparable):
    # override the default cache object
    __cache__ = WeakCache()
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


def pair(a, b):
    # for same ordering with comparable
    if id(a) < id(b):
        return (a, b)
    else:
        return (b, a)


def test_comparable_basic(cache):
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

    cache[pair(a, b)] = True
    cache[pair(a, c)] = False
    cache[pair(c, d)] = True
    cache[pair(b, d)] = False
    assert len(cache) == 4

    assert a == b
    assert a != c
    assert c == d
    assert b != d
    assert Node.num_equal_calls == 0

    # no cache hit
    assert pair(a, e) not in cache
    assert a != e
    assert Node.num_equal_calls == 1
    assert len(cache) == 5

    # run only once
    assert e != a
    assert Node.num_equal_calls == 1
    assert pair(a, e) in cache


def test_comparable_garbage_collection(cache):
    a = Node(name="a")
    b = Node(name="b")
    c = Node(name="c")
    d = Node(name="d")

    cache[pair(a, b)] = True
    cache[pair(a, c)] = False
    cache[pair(c, d)] = True
    cache[pair(b, d)] = False

    assert weakref.getweakrefcount(a) == 2
    del c
    assert weakref.getweakrefcount(a) == 1
    del b
    assert weakref.getweakrefcount(a) == 0


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

    def __init__(self, a, b):
        super().__init__(a=a, b=b)


def test_slotted():
    obj = MyObj(1, 2)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.__slots__ == ("a", "b")
    with pytest.raises(AttributeError):
        obj.c = 3

    obj2 = MyObj(1, 2)
    assert obj == obj2
    assert obj is not obj2

    obj3 = MyObj(1, 3)
    assert obj != obj3

    assert pickle.loads(pickle.dumps(obj)) == obj


class MyFrozenObj(FrozenSlotted):
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        super().__init__(a=a, b=b)


def test_frozen_slotted():
    obj = MyFrozenObj(1, 2)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.__slots__ == ("a", "b")
    with pytest.raises(AttributeError):
        obj.b = 3
    with pytest.raises(AttributeError):
        obj.c = 3

    obj2 = MyFrozenObj(1, 2)
    assert obj == obj2
    assert obj is not obj2
    assert hash(obj) == hash(obj2)

    restored = pickle.loads(pickle.dumps(obj))
    assert restored == obj
    assert hash(restored) == hash(obj)
