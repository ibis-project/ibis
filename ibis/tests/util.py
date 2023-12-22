"""Test utilities."""

from __future__ import annotations

import pickle
from typing import Callable

import ibis
import ibis.expr.types as ir
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


def schemas_eq(left: ir.Expr, right: ir.Expr) -> bool:
    left_schema = left.as_table().schema()
    right_schema = right.as_table().schema()
    return left_schema == right_schema


def assert_decompile_roundtrip(
    expr: ir.Expr,
    snapshot=None,
    eq: Callable[[ir.Expr, ir.Expr], bool] = ir.Expr.equals,
):
    """Assert that an ibis expression remains the same after decompilation.

    Parameters
    ----------
    expr
        The expression to decompile.
    snapshot
        A snapshot fixture.
    eq
        A callable that returns whether two Ibis expressions are equal.
        Defaults to `ibis.expr.types.Expr.equals`. Use this to adjust
        comparison behavior for expressions that contain `SelfReference`
        operations from table.view() calls, or other relations whose equality
        is difficult to roundtrip.
    """
    rendered = ibis.decompile(expr, format=True)
    if snapshot is not None:
        snapshot.assert_match(rendered, "decompiled.py")

    # execute the rendered python code
    locals_ = {}
    exec(rendered, {}, locals_)
    restored = locals_["result"]

    assert eq(expr.unbind(), restored)
