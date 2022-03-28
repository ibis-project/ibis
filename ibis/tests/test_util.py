"""Test ibis.util utilities."""


import weakref

import pytest

from ibis import util


class Node(util.Comparable):

    # override the default cache object
    __cache__ = util.WeakCache()
    num_equal_calls = 0

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Node(name={self.name})"

    def __equals__(self, other):
        Node.num_equal_calls += 1
        return self.name == other.name


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ([], []),
        ([1], [1]),
        ([1, 2], [1, 2]),
        ([[]], []),
        ([[1]], [1]),
        ([[1, 2]], [1, 2]),
        ([[1, [2, [3, [4]]]]], [1, 2, 3, 4]),
        ([[4, [3, [2, [1]]]]], [4, 3, 2, 1]),
        ([[[[4], 3], 2], 1], [4, 3, 2, 1]),
        ([[[[1], 2], 3], 4], [1, 2, 3, 4]),
        ([{(1,), frozenset({(2,)})}], {1, 2}),
        ({(1, (2,)): None, "a": None}, [1, 2, "a"]),
        (([x] for x in range(5)), list(range(5))),
        ({(1, (2, frozenset({(3,)})))}, [1, 2, 3]),
    ],
)
def test_flatten(case, expected):
    assert type(expected)(util.flatten_iterable(case)) == expected


@pytest.mark.parametrize("case", [1, "abc", b"abc", 2.0, object()])
def test_flatten_invalid_input(case):
    flat = util.flatten_iterable(case)

    with pytest.raises(TypeError):
        list(flat)


def test_toposort_empty_graph():
    assert not list(util.toposort({}))


def test_toposort_simple():
    dag = {1: [2, 3], 2: [3], 3: []}
    assert list(util.toposort(dag)) == [3, 2, 1]


def test_toposort_cycle():
    # 1 depends on itself
    dag = {1: [1, 2, 3], 2: [3], 3: []}
    with pytest.raises(ValueError):
        list(util.toposort(dag))


def test_toposort_missing_key():
    # 3 is a dependency but not a key
    dag = {1: [1, 2, 3], 2: [3]}
    with pytest.raises(KeyError):
        list(util.toposort(dag))


@pytest.fixture
def cache():
    Node.num_equal_calls = 0
    cache = Node.__cache__
    try:
        yield cache
    finally:
        assert len(cache) == 0


def pair(a, b):
    # for same ordering with comparable
    if hash(a) < hash(b):
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
        a == a
        a == b
        b == a
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
