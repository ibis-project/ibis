"""Test ibis.util utilities."""


import pytest

from ibis import util
from ibis.tests.util import assert_pickle_roundtrip


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


def test_dotdict():
    d = util.DotDict({"a": 1, "b": 2, "c": 3})
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
    d = util.frozendict({"a": 1, "b": 2, "c": 3})
    e = util.frozendict(a=1, b=2, c=3)
    assert d == e
    assert d["a"] == 1
    assert d["b"] == 2

    msg = "'frozendict' object does not support item assignment"
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


def test_import_object():
    import collections

    assert util.import_object("collections.defaultdict") is collections.defaultdict
    assert util.import_object("collections.abc.Mapping") is collections.abc.Mapping

    with pytest.raises(ImportError):
        util.import_object("this_module_probably.doesnt_exist")

    with pytest.raises(ImportError):
        util.import_object("collections.this_attribute_doesnt_exist")
