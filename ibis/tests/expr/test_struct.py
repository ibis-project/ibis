from __future__ import annotations

from collections import OrderedDict

import pytest

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _
from ibis.expr.tests.test_newrels import join_tables
from ibis.tests.util import assert_pickle_roundtrip


@pytest.fixture
def t():
    return ibis.table(dict(a="struct<b: float, c: string>", d="string"), name="t")


@pytest.fixture
def s():
    return ibis.table(dict(a="struct<f: float, g: string>"), name="s")


def test_struct_operations():
    value = OrderedDict(
        [
            ("a", 1),
            ("b", list("abc")),
            ("c", OrderedDict([("foo", [1.0, 2.0])])),
        ]
    )
    expr = ibis.literal(value)
    assert isinstance(expr, ir.StructScalar)
    assert isinstance(expr["b"], ir.ArrayScalar)
    assert isinstance(expr["a"], ir.IntegerScalar)


def test_struct_getattr():
    expr = ibis.struct({"a": 1, "b": 2})
    assert isinstance(expr.a, ir.IntegerValue)
    with pytest.raises(AttributeError, match="bad"):
        expr.bad  # # noqa: B018


def test_null_literal_getitem():
    expr = ibis.literal(None, type="struct<a: int64, b: string>")
    field = expr["a"]
    assert isinstance(field, ir.IntegerValue)
    assert isinstance(field.op(), ops.Literal)
    assert field.op().value is None


def test_struct_tab_completion():
    t = ibis.table([("struct_col", "struct<my_field: string, for: int64>")])
    # Only valid python identifiers in getattr completions
    attrs = dir(t.struct_col)
    assert "my_field" in attrs
    assert "for" not in attrs
    # All fields in getitem completions
    items = t.struct_col._ipython_key_completions_()
    assert {"my_field", "for"}.issubset(items)


def test_struct_pickle():
    struct_scalar_expr = ibis.literal(OrderedDict([("fruit", "pear"), ("weight", 0)]))

    assert_pickle_roundtrip(struct_scalar_expr)


def test_lift(t):
    assert t.a.lift().equals(t.select(_.a.b, _.a.c))


def test_unpack_from_table(t):
    assert t.unpack("a").equals(t.select(_.a.b, _.a.c, _.d))


def test_lift_join(t, s):
    join = t.join(s, t.d == s.a.g)
    result = join.a_right.lift()
    with join_tables(join) as (tt, ss):
        expected = ops.JoinChain(
            first=tt,
            rest=[
                ops.JoinLink("inner", ss, [tt.d == ss.a.g]),
            ],
            values={"f": ss.a.f, "g": ss.a.g},
        )
        assert result.op() == expected


def test_unpack_join_from_table(t, s):
    join = t.join(s, t.d == s.a.g)
    result = join.unpack("a_right")
    expected = join.select(_.a, _.d, _.a_right.f, _.a_right.g)
    assert result.equals(expected)


def test_nested_lift():
    t = ibis.table(
        {"a": "struct<b:struct<x: int, y: int>, c: string>", "d": "string"},
        name="t",
    )
    expr = t.a.b.lift()
    assert expr.schema() == ibis.schema({"x": "int", "y": "int"})
