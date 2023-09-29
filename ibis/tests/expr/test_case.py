from __future__ import annotations

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _
from ibis.tests.util import assert_equal, assert_pickle_roundtrip


def test_ifelse_method(table):
    bools = table.g.isnull()
    result = bools.ifelse("foo", "bar")
    assert isinstance(result, ir.StringColumn)


def test_ifelse_function_literals():
    res = ibis.ifelse(True, 1, 2)
    sol = ibis.literal(True, type="bool").ifelse(1, 2)
    assert res.equals(sol)

    # condition is explicitly cast to bool
    res = ibis.ifelse(1, 1, 2)
    sol = ibis.literal(1, type="bool").ifelse(1, 2)
    assert res.equals(sol)


def test_ifelse_function_exprs(table):
    res = ibis.ifelse(table.g.isnull(), 1, table.a)
    sol = table.g.isnull().ifelse(1, table.a)
    assert res.equals(sol)

    # condition is cast if not already bool
    res = ibis.ifelse(table.a, 1, table.b)
    sol = table.a.cast("bool").ifelse(1, table.b)
    assert res.equals(sol)


def test_ifelse_function_deferred(table):
    expr = ibis.ifelse(_.g.isnull(), _.a, 2)
    assert repr(expr) == "ifelse(_.g.isnull(), _.a, 2)"
    res = expr.resolve(table)
    sol = table.g.isnull().ifelse(table.a, 2)
    assert res.equals(sol)


def test_simple_case_expr(table):
    case1, result1 = "foo", table.a
    case2, result2 = "bar", table.c
    default_result = table.b

    expr1 = table.g.lower().cases(
        [(case1, result1), (case2, result2)], default=default_result
    )

    expr2 = (
        table.g.lower()
        .case()
        .when(case1, result1)
        .when(case2, result2)
        .else_(default_result)
        .end()
    )

    assert_equal(expr1, expr2)
    assert isinstance(expr1, ir.IntegerColumn)


def test_multiple_case_expr(table):
    expr = (
        ibis.case()
        .when(table.a == 5, table.f)
        .when(table.b == 128, table.b * 2)
        .when(table.c == 1000, table.e)
        .else_(table.d)
        .end()
    )

    # deferred cases
    deferred = (
        ibis.case()
        .when(_.a == 5, table.f)
        .when(_.b == 128, table.b * 2)
        .when(_.c == 1000, table.e)
        .else_(table.d)
        .end()
    )
    expr2 = deferred.resolve(table)

    # deferred results
    expr3 = (
        ibis.case()
        .when(table.a == 5, _.f)
        .when(table.b == 128, _.b * 2)
        .when(table.c == 1000, _.e)
        .else_(table.d)
        .end()
        .resolve(table)
    )

    # deferred default
    expr4 = (
        ibis.case()
        .when(table.a == 5, table.f)
        .when(table.b == 128, table.b * 2)
        .when(table.c == 1000, table.e)
        .else_(_.d)
        .end()
        .resolve(table)
    )

    assert repr(deferred) == "<case>"
    assert expr.equals(expr2)
    assert expr.equals(expr3)
    assert expr.equals(expr4)

    op = expr.op()
    assert isinstance(expr, ir.FloatingColumn)
    assert isinstance(op, ops.SearchedCase)
    assert op.default == table.d.op()


def test_pickle_multiple_case_node(table):
    case1 = table.a == 5
    case2 = table.b == 128
    case3 = table.c == 1000

    result1 = table.f
    result2 = table.b * 2
    result3 = table.e

    default = table.d
    expr = (
        ibis.case()
        .when(case1, result1)
        .when(case2, result2)
        .when(case3, result3)
        .else_(default)
        .end()
    )

    op = expr.op()
    assert_pickle_roundtrip(op)


def test_simple_case_null_else(table):
    expr = table.g.case().when("foo", "bar").end()
    op = expr.op()

    assert isinstance(expr, ir.StringColumn)
    assert isinstance(op.default.to_expr(), ir.Value)
    assert isinstance(op.default, ops.Cast)
    assert op.default.to == dt.string


def test_multiple_case_null_else(table):
    expr = ibis.case().when(table.g == "foo", "bar").end()
    expr2 = ibis.case().when(table.g == "foo", _).end().resolve("bar")

    assert expr.equals(expr2)

    op = expr.op()
    assert isinstance(expr, ir.StringColumn)
    assert isinstance(op.default.to_expr(), ir.Value)
    assert isinstance(op.default, ops.Cast)
    assert op.default.to == dt.string


def test_case_mixed_type():
    t0 = ibis.table(
        [("one", "string"), ("two", "double"), ("three", "int32")],
        name="my_data",
    )

    expr = (
        t0.three.case().when(0, "low").when(1, "high").else_("null").end().name("label")
    )
    result = t0[expr]
    assert result["label"].type().equals(dt.string)
