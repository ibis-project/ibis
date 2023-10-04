"""Test ibis.util utilities."""
from __future__ import annotations

import pytest

from ibis.util import PseudoHashable, flatten_iterable, import_object


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
    assert type(expected)(flatten_iterable(case)) == expected


@pytest.mark.parametrize("case", [1, "abc", b"abc", 2.0, object()])
def test_flatten_invalid_input(case):
    flat = flatten_iterable(case)

    with pytest.raises(TypeError):
        list(flat)


def test_import_object():
    import collections

    assert import_object("collections.defaultdict") is collections.defaultdict
    assert import_object("collections.abc.Mapping") is collections.abc.Mapping

    with pytest.raises(ImportError):
        import_object("this_module_probably.doesnt_exist")

    with pytest.raises(ImportError):
        import_object("collections.this_attribute_doesnt_exist")


# TODO(kszucs): add tests for promote_list and promote_tuple


def test_pseudo_hashable():
    class Unhashable:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Unhashable) and self.value == other.value

    class MyList(list):
        pass

    class MyMap(dict):
        pass

    for obj in [1, "a", b"a", 2.0, object(), (), frozenset()]:
        with pytest.raises(TypeError, match="Cannot wrap a hashable object"):
            PseudoHashable(obj)

    # test unhashable sequences
    lst1 = [1, 2, 3]
    lst2 = [1, 2, 4]
    lst3 = MyList([1, 2, 3])
    ph1 = PseudoHashable(lst1)
    ph2 = PseudoHashable(lst2)
    ph3 = PseudoHashable(lst3)
    ph4 = PseudoHashable(lst1.copy())

    assert hash(ph1) == hash(ph1)
    assert hash(ph1) != hash(ph2)
    assert hash(ph1) != hash(ph3)
    assert hash(ph2) != hash(ph3)
    assert hash(ph3) == hash(ph3)
    assert hash(ph1) == hash(ph4)
    assert ph1 == ph1
    assert ph1 != ph2
    assert ph1 == ph3
    assert ph2 != ph3
    assert ph3 == ph3
    assert ph1 == ph4

    # test unhashable mappings
    dct1 = {"a": 1, "b": 2}
    dct2 = {"a": 1, "b": 3}
    dct3 = MyMap({"a": 1, "b": 2})
    ph1 = PseudoHashable(dct1)
    ph2 = PseudoHashable(dct2)
    ph3 = PseudoHashable(dct3)
    ph4 = PseudoHashable(dct1.copy())

    assert hash(ph1) == hash(ph1)
    assert hash(ph1) != hash(ph2)
    assert hash(ph1) != hash(ph3)
    assert hash(ph2) != hash(ph3)
    assert hash(ph3) == hash(ph3)
    assert hash(ph1) == hash(ph4)
    assert ph1 == ph1
    assert ph1 != ph2
    assert ph1 == ph3
    assert ph2 != ph3
    assert ph3 == ph3
    assert ph1 == ph4

    # test unhashable objects
    obj1 = Unhashable(1)
    obj2 = Unhashable(1)
    obj3 = Unhashable(2)
    obj4 = Unhashable(1)
    ph1 = PseudoHashable(obj1)
    ph2 = PseudoHashable(obj2)
    ph3 = PseudoHashable(obj3)
    ph4 = PseudoHashable(obj4)

    assert hash(ph1) == hash(ph1)
    assert hash(ph1) != hash(ph2)
    assert hash(ph1) != hash(ph3)
    assert hash(ph2) != hash(ph3)
    assert hash(ph3) == hash(ph3)
    assert hash(ph1) != hash(ph4)
    assert ph1 == ph1
    assert ph1 == ph2
    assert ph1 != ph3
    assert ph2 != ph3
    assert ph3 == ph3
    assert ph1 == ph4
