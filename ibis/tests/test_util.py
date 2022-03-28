"""Test ibis.util utilities."""


import pytest

from ibis import util


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
