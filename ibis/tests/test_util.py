"""Test ibis.util utilities."""
from __future__ import annotations

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


def test_import_object():
    import collections

    assert util.import_object("collections.defaultdict") is collections.defaultdict
    assert util.import_object("collections.abc.Mapping") is collections.abc.Mapping

    with pytest.raises(ImportError):
        util.import_object("this_module_probably.doesnt_exist")

    with pytest.raises(ImportError):
        util.import_object("collections.this_attribute_doesnt_exist")


# TODO(kszucs): add tests for promote_list and promote_tuple
