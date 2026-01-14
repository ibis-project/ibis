"""Test ibis.util utilities."""

from __future__ import annotations

import pytest

from ibis.util import (
    PseudoHashable,
    flatten_iterable,
    import_object,
    promote_list,
    promote_tuple,
)


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


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        # Already a list - pass through unchanged
        pytest.param([], [], id="empty_list"),
        pytest.param([1, 2, 3], [1, 2, 3], id="list_of_ints"),
        pytest.param([[1], [2]], [[1], [2]], id="nested_list"),
        # Dictionary - wrap in list (special case)
        pytest.param({}, [{}], id="empty_dict"),
        pytest.param({"a": 1}, [{"a": 1}], id="dict"),
        # Other iterables - convert to list
        pytest.param((), [], id="empty_tuple"),
        pytest.param((1, 2, 3), [1, 2, 3], id="tuple"),
        pytest.param(set(), [], id="empty_set"),
        pytest.param(frozenset({1, 2}), [1, 2], id="frozenset"),
        pytest.param(range(3), [0, 1, 2], id="range"),
        # None - return empty list
        pytest.param(None, [], id="none"),
        # Non-iterables - wrap in list
        pytest.param(42, [42], id="int"),
        pytest.param(3.14, [3.14], id="float"),
        pytest.param(True, [True], id="bool"),
        # Strings and bytes are treated as non-iterable
        pytest.param("hello", ["hello"], id="string"),
        pytest.param(b"bytes", [b"bytes"], id="bytes"),
    ],
)
def test_promote_list(val, expected):
    result = promote_list(val)
    # For sets/frozensets, order is not guaranteed
    if isinstance(val, (set, frozenset)):
        assert sorted(result) == sorted(expected)
    else:
        assert result == expected


def test_promote_list_identity():
    """Test that lists are returned as-is (same object)."""
    val = [1, 2, 3]
    assert promote_list(val) is val


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        # Already a tuple - pass through unchanged
        pytest.param((), (), id="empty_tuple"),
        pytest.param((1, 2, 3), (1, 2, 3), id="tuple_of_ints"),
        pytest.param(((1,), (2,)), ((1,), (2,)), id="nested_tuple"),
        # Other iterables - convert to tuple
        pytest.param([], (), id="empty_list"),
        pytest.param([1, 2, 3], (1, 2, 3), id="list"),
        pytest.param(set(), (), id="empty_set"),
        pytest.param(frozenset({1, 2}), (1, 2), id="frozenset"),
        pytest.param(range(3), (0, 1, 2), id="range"),
        # Dict - iterates over keys
        pytest.param({}, (), id="empty_dict"),
        pytest.param({"a": 1, "b": 2}, ("a", "b"), id="dict_keys"),
        # None - return empty tuple
        pytest.param(None, (), id="none"),
        # Non-iterables - wrap in tuple
        pytest.param(42, (42,), id="int"),
        pytest.param(3.14, (3.14,), id="float"),
        pytest.param(True, (True,), id="bool"),
        # Strings and bytes are treated as non-iterable
        pytest.param("hello", ("hello",), id="string"),
        pytest.param(b"bytes", (b"bytes",), id="bytes"),
    ],
)
def test_promote_tuple(val, expected):
    result = promote_tuple(val)
    # For sets/frozensets, order is not guaranteed
    if isinstance(val, (set, frozenset)):
        assert tuple(sorted(result)) == tuple(sorted(expected))
    else:
        assert result == expected


def test_promote_tuple_identity():
    """Test that tuples are returned as-is (same object)."""
    val = (1, 2, 3)
    assert promote_tuple(val) is val


def test_pseudo_hashable():
    class Unhashable:  # noqa: PLW1641
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
