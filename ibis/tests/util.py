"""Test utilities."""

from __future__ import annotations

import pickle

import ibis
from ibis import util


def assert_equal(left, right):
    """Assert that two ibis objects are equal."""

    if util.all_of([left, right], ibis.Schema):
        assert left.equals(right), f"Comparing schemas: \n{left!r} !=\n{right!r}"
    else:
        assert left.equals(right), f"Objects unequal: \n{left!r}\nvs\n{right!r}"


def assert_pickle_roundtrip(obj):
    """Assert that an ibis object remains the same after pickling and
    unpickling."""
    loaded = pickle.loads(pickle.dumps(obj))
    if hasattr(obj, "equals"):
        assert obj.equals(loaded)
    else:
        assert obj == loaded


def assert_decompile_roundtrip(expr, snapshot=None, check_equality=True):
    """Assert that an ibis expression remains the same after decompilation."""
    rendered = ibis.decompile(expr, format=True)
    if snapshot is not None:
        snapshot.assert_match(rendered, "decompiled.py")

    # execute the rendered python code
    locals_ = {}
    exec(rendered, {}, locals_)
    restored = locals_["result"]

    if check_equality:
        assert expr.unbind().equals(restored)
    else:
        assert expr.as_table().schema().equals(restored.as_table().schema())
