from __future__ import annotations

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _
from ibis.common.annotations import SignatureValidationError
from ibis.tests.util import assert_pickle_roundtrip


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


def test_case_dshape(table):
    assert isinstance(ibis.cases((True, "bar"), (False, "bar")), ir.Scalar)
    assert isinstance(ibis.cases((True, None), else_="bar"), ir.Scalar)
    assert isinstance(ibis.cases((table.b == 9, None), else_="bar"), ir.Column)
    assert isinstance(ibis.cases((True, table.a), else_=42), ir.Column)
    assert isinstance(ibis.cases((True, 42), else_=table.a), ir.Column)
    assert isinstance(ibis.cases((True, table.a), else_=table.b), ir.Column)

    assert isinstance(ibis.literal(5).cases((9, 42)), ir.Scalar)
    assert isinstance(ibis.literal(5).cases((9, 42), else_=43), ir.Scalar)
    assert isinstance(ibis.literal(5).cases((table.a, 42)), ir.Column)
    assert isinstance(ibis.literal(5).cases((9, table.a)), ir.Column)
    assert isinstance(ibis.literal(5).cases((table.a, table.b)), ir.Column)
    assert isinstance(ibis.literal(5).cases((9, 42), else_=table.a), ir.Column)
    assert isinstance(table.a.cases((9, 42)), ir.Column)
    assert isinstance(table.a.cases((table.b, 42)), ir.Column)
    assert isinstance(table.a.cases((9, table.b)), ir.Column)
    assert isinstance(table.a.cases((table.a, table.b)), ir.Column)


def test_case_dtype():
    assert isinstance(ibis.cases((True, "bar"), (False, "bar")), ir.StringValue)
    assert isinstance(ibis.cases((True, None), else_="bar"), ir.StringValue)
    with pytest.raises(TypeError):
        ibis.cases((True, 5), (False, "bar"))
    with pytest.raises(TypeError):
        ibis.cases((True, 5), else_="bar")


def test_multiple_case_expr(table):
    expr = ibis.cases(
        (table.a == 5, table.f),
        (table.b == 128, table.b * 2),
        (table.c == 1000, table.e),
        else_=table.d,
    )

    # deferred cases
    deferred = ibis.cases(
        (_.a == 5, table.f),
        (_.b == 128, table.b * 2),
        (_.c == 1000, table.e),
        else_=table.d,
    )
    expr2 = deferred.resolve(table)

    # deferred results
    expr3 = ibis.cases(
        (table.a == 5, _.f),
        (table.b == 128, _.b * 2),
        (table.c == 1000, _.e),
        else_=table.d,
    ).resolve(table)

    # deferred default
    expr4 = ibis.cases(
        (table.a == 5, table.f),
        (table.b == 128, table.b * 2),
        (table.c == 1000, table.e),
        else_=_.d,
    ).resolve(table)

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
    expr = ibis.cases(
        (case1, result1),
        (case2, result2),
        (case3, result3),
        else_=default,
    )

    op = expr.op()
    assert_pickle_roundtrip(op)


def test_simple_case_null_else(table):
    expr = table.g.cases(("foo", "bar"))
    op = expr.op()

    assert isinstance(expr, ir.StringColumn)
    assert isinstance(op.default.to_expr(), ir.Value)


def test_multiple_case_null_else(table):
    expr = ibis.cases((table.g == "foo", "bar"))
    expr2 = ibis.cases((table.g == "foo", _)).resolve("bar")

    assert expr.equals(expr2)

    op = expr.op()
    assert isinstance(expr, ir.StringColumn)
    assert isinstance(op.default.to_expr(), ir.Value)


def test_case_mixed_type():
    t0 = ibis.table(
        [("one", "string"), ("two", "double"), ("three", "int32")],
        name="my_data",
    )

    expr = t0.three.cases((0, "low"), (1, "high"), else_="null").name("label")
    result = t0.select(expr)
    assert result["label"].type().equals(dt.string)


def test_err_on_bad_args():
    with pytest.raises(ValueError):
        ibis.cases((True,))
    with pytest.raises(ValueError):
        ibis.cases((True, 3, 4))
    with pytest.raises(ValueError):
        ibis.cases((True, 3, 4))
    with pytest.raises(TypeError):
        ibis.cases((True, 3), 5)


def test_err_on_nonbool_expr(table):
    with pytest.raises(SignatureValidationError):
        ibis.cases((table.a, "bar"), else_="baz")
    with pytest.raises(SignatureValidationError):
        ibis.cases((ibis.literal(1), "bar"), else_=("baz"))


def test_err_on_noncomparable(table):
    table.a.cases((8, "bar"))
    table.a.cases((-8, "bar"))
    # Can't compare an int to a string
    with pytest.raises(TypeError):
        table.a.cases(("foo", "bar"))


def test_err_on_empty_cases(table):
    with pytest.raises(TypeError):
        ibis.cases()
    with pytest.raises(TypeError):
        ibis.cases(else_=42)
    with pytest.raises(TypeError):
        table.a.cases()
    with pytest.raises(TypeError):
        table.a.cases(else_=42)
