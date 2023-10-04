from __future__ import annotations

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.deferred import _, const, deferred, var
from ibis.common.patterns import AnyOf, Check, pattern, replace
from ibis.util import Namespace

p = Namespace(pattern, module=ops)
d = Namespace(deferred, module=ops)

x = var("x")
y = var("y")
z = var("z")
params = var("params")

zero = ibis.literal(0)
one = ibis.literal(1)
two = ibis.literal(2)
three = ibis.literal(3)
add_1_2 = one + two

param_a = ibis.param("int8")
param_b = ibis.param("int8")
param_values = {param_a.op(): 1, param_b.op(): 2}

get_literal_value = p.Literal >> _.value
inc_integer_literal = p.Literal(x, dtype=dt.Integer) >> _.copy(value=x + 1)
sub_param_from_const = p.ScalarParameter >> d.Literal(
    const(param_values)[_], dtype=_.dtype
)


@replace(p.Add(p.Literal(x), p.Literal(y)))
def fold_literal_add(_, x, y):
    return ibis.literal(x + y).op()


simplifications = AnyOf(
    p.Add(x, p.Literal(0)) >> x,
    p.Add(p.Literal(0), x) >> x,
    p.Subtract(x, p.Literal(0)) >> x,
    p.Multiply(x, p.Literal(1)) >> x,
    p.Multiply(p.Literal(1), x) >> x,
    p.Multiply(x, p.Literal(0)) >> zero.op(),
    p.Multiply(p.Literal(0), x) >> zero.op(),
    p.Divide(x, p.Literal(1)) >> x,
    p.Divide(p.Literal(0), x) >> zero.op(),
    p.Divide(x, y) & Check(x == y) >> one.op(),
    fold_literal_add,
)


@replace(p.Literal(value=x, dtype=y))
def literal_to_type_call(_, x, y):
    return f"{y}({x})"


@pytest.mark.parametrize(
    ("rule", "expr", "expected"),
    [
        (get_literal_value, one, 1),
        (get_literal_value, two, 2),
        (literal_to_type_call, one, "int8(1)"),
        (inc_integer_literal, one, two.op()),
        (sub_param_from_const, param_a + param_b, add_1_2.op()),
    ],
)
def test_replace_scalar_parameters(rule, expr, expected):
    assert expr.op().replace(rule) == expected


def test_replace_scalar_parameters_using_variable():
    expr = param_a + param_b
    context = {"params": param_values}
    sub_param_from_var = p.ScalarParameter(x) >> d.Literal(params[_], dtype=x)
    assert expr.op().replace(sub_param_from_var, context=context) == add_1_2.op()


def test_replace_propagation():
    expr = add_1_2 + add_1_2 + add_1_2
    rule = p.Add(p.Add(x, y), z) >> d.Subtract(d.Subtract(x, y), z)
    result = expr.op().replace(rule)
    expected = ((one - two) - add_1_2) + add_1_2
    assert result == expected.op()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (one + zero, one),
        (zero + one, one),
        (one - zero, one),
        (one * zero, zero),
        (zero * one, zero),
        ((one + one + one) * one, three),
        (one / one, one),
        (one / one / one, one),
        (one / (one / one), one),
        (one / (one / one) / one, one),
        (three / (one / one) / one, three),
        (three / three, one),
    ],
)
def test_simplification(expr, expected):
    result = expr.op().replace(simplifications)
    assert result == expected.op()
