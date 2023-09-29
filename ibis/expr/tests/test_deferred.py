from __future__ import annotations

import operator
import pickle

import pytest
from pytest import param

import ibis
from ibis import _
from ibis.expr.deferred import DeferredCall, deferrable


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

    with pytest.raises(AttributeError, match="DeferredAttr.+__fizzbuzz__"):
        _.a.__fizzbuzz__()


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


def test_deferred_call(table):
    expr = DeferredCall(operator.add, (_.a, 2))
    res = expr.resolve(table)
    assert res.equals(table.a + 2)
    assert repr(expr) == "add(_.a, 2)"

    func = lambda a, b: a + b
    expr = DeferredCall(func, kwargs=dict(a=_.a, b=2))
    res = expr.resolve(table)
    assert res.equals(table.a + 2)
    assert func.__name__ in repr(expr)
    assert "a=_.a, b=2" in repr(expr)

    expr = DeferredCall(operator.add, (_.a, 2), repr="<test>")
    assert repr(expr) == "<test>"


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


@pytest.mark.parametrize("obj", [_, _.a, _.a.b[0]])
def test_deferred_is_not_iterable(obj):
    with pytest.raises(TypeError, match="object is not iterable"):
        sorted(obj)

    with pytest.raises(TypeError, match="object is not iterable"):
        iter(obj)

    with pytest.raises(TypeError, match="is not an iterator"):
        next(obj)


def test_deferrable(table):
    @deferrable
    def f(a, b, c=3):
        return a + b + c

    assert f(table.a, table.b).equals(table.a + table.b + 3)
    assert f(table.a, table.b, c=4).equals(table.a + table.b + 4)

    expr = f(_.a, _.b)
    sol = table.a + table.b + 3
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == "f(_.a, _.b)"

    expr = f(1, 2, c=_.a)
    sol = 3 + table.a
    res = expr.resolve(table)
    assert res.equals(sol)
    assert repr(expr) == "f(1, 2, c=_.a)"

    with pytest.raises(TypeError, match="unknown"):
        f(_.a, _.b, unknown=3)  # invalid calls caught at call time


@pytest.mark.parametrize(
    "case",
    [
        param(lambda: ([1, _], [1, 2]), id="list"),
        param(lambda: ((1, _), (1, 2)), id="tuple"),
        param(lambda: ({1, _}, {1, 2}), id="set"),
        param(lambda: ({"x": 1, "y": _}, {"x": 1, "y": 2}), id="dict"),
        param(lambda: ({"x": 1, "y": [_, 3]}, {"x": 1, "y": [2, 3]}), id="nested"),
    ],
)
def test_deferrable_nested_args(case):
    arg, sol = case()

    @deferrable
    def identity(x):
        return x

    expr = identity(arg)
    assert expr.resolve(2) == sol
    assert identity(sol) is sol
    assert repr(expr) == f"identity({arg!r})"


def test_deferrable_repr():
    @deferrable(repr="<test>")
    def myfunc(x):
        return x + 1

    assert repr(myfunc(_.a)) == "<test>"


@pytest.mark.parametrize(
    "f, sol",
    [
        (lambda t: _.x + t.a, "(_.x + <column[int64]>)"),
        (lambda t: _.x + t.a.sum(), "(_.x + <scalar[int64]>)"),
        (lambda t: ibis.date(_.x, 2, t.a), "date(_.x, 2, <column[int64]>)"),
    ],
)
def test_repr_deferred_with_exprs(f, sol):
    t = ibis.table({"a": "int64"})
    expr = f(t)
    res = repr(expr)
    assert res == sol
