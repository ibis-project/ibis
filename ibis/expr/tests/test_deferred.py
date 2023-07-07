from __future__ import annotations

import operator
import pickle

import pytest
from pytest import param

import ibis
from ibis import _


@pytest.fixture
def table():
    return ibis.table({"a": "int", "b": "int", "c": "str"})


def test_hash():
    """Deferred expressions must be hashable"""
    expr1 = _.a
    expr2 = _.a + 1
    assert hash(expr1) == hash(expr1)
    assert hash(expr1) != hash(expr2)


@pytest.mark.parametrize(
    "func",
    [
        param(lambda _: _, id="root"),
        param(lambda _: _.a, id="getattr"),
        param(lambda _: _["a"], id="getitem"),
        param(lambda _: _.a.log(), id="method"),
        param(lambda _: _.a.log(_.b), id="method-with-args"),
        param(lambda _: _.a.log(base=_.b), id="method-with-kwargs"),
        param(lambda _: _.a + _.b, id="binary-op"),
        param(lambda _: ~_.a, id="unary-op"),
    ],
)
def test_pickle(func, table):
    expr1 = func(_)
    expr2 = pickle.loads(pickle.dumps(expr1))

    r1 = expr1.resolve(table)
    r2 = expr2.resolve(table)
    assert r1.equals(r2)


def test_magic_methods_not_deferred():
    with pytest.raises(AttributeError, match="__fizzbuzz__"):
        _.__fizzbuzz__()


def test_getattr(table):
    expr = _.a
    sol = table.a
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == "_.a"


def test_getitem(table):
    expr = _["a"]
    sol = table["a"]
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == "_['a']"


def test_method(table):
    expr = _.a.log()
    res = expr.resolve(table)
    assert res.equals(table.a.log())
    assert repr(expr) == "_.a.log()"


def test_method_args(table):
    expr = _.a.log(1)
    res = expr.resolve(table)
    assert res.equals(table.a.log(1))
    assert repr(expr) == "_.a.log(1)"

    expr = _.a.log(_.b)
    res = expr.resolve(table)
    assert res.equals(table.a.log(table.b))
    assert repr(expr) == "_.a.log(_.b)"


def test_method_kwargs(table):
    expr = _.a.log(base=1)
    res = expr.resolve(table)
    assert res.equals(table.a.log(base=1))
    assert repr(expr) == "_.a.log(base=1)"

    expr = _.a.log(base=_.b)
    res = expr.resolve(table)
    assert res.equals(table.a.log(base=table.b))
    assert repr(expr) == "_.a.log(base=_.b)"


@pytest.mark.parametrize(
    "symbol, op",
    [
        ("+", operator.add),
        ("-", operator.sub),
        ("*", operator.mul),
        ("/", operator.truediv),
        ("//", operator.floordiv),
        ("**", operator.pow),
        ("%", operator.mod),
        ("&", operator.and_),
        ("|", operator.or_),
        ("^", operator.xor),
        (">>", operator.rshift),
        ("<<", operator.lshift),
    ],
)
def test_binary_ops(symbol, op, table):
    expr = op(_.a, _.b)
    sol = op(table.a, table.b)
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == f"(_.a {symbol} _.b)"

    expr = op(1, _.a)
    sol = op(1, table.a)
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == f"(1 {symbol} _.a)"


@pytest.mark.parametrize(
    "sym, rsym, op",
    [
        ("==", "==", operator.eq),
        ("!=", "!=", operator.ne),
        ("<", ">", operator.lt),
        ("<=", ">=", operator.le),
        (">", "<", operator.gt),
        (">=", "<=", operator.ge),
    ],
)
def test_compare_ops(sym, rsym, op, table):
    expr = op(_.a, _.b)
    sol = op(table.a, table.b)
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == f"(_.a {sym} _.b)"

    expr = op(1, _.a)
    sol = op(1, table.a)
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == f"(_.a {rsym} 1)"


@pytest.mark.parametrize(
    "symbol, op",
    [
        ("-", operator.neg),
        ("~", operator.invert),
    ],
)
def test_unary_ops(symbol, op, table):
    expr = op(_.a)
    sol = op(table.a)
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == f"{symbol}_.a"
