from __future__ import annotations

import operator
import pickle

import pytest
from pytest import param

from ibis.common.bases import Slotted
from ibis.common.collections import FrozenDict
from ibis.common.deferred import (
    Attr,
    Call,
    Deferred,
    Factory,
    Item,
    Just,
    JustUnhashable,
    Mapping,
    Sequence,
    Variable,
    _,
    const,
    deferrable,
    deferred,
    resolver,
    var,
)
from ibis.util import Namespace


def test_builder_just():
    p = Just(1)
    assert p.resolve({}) == 1
    assert p.resolve({"a": 1}) == 1

    # unwrap subsequently nested Just instances
    assert Just(p) == p

    # disallow creating a Just builder from other builders or deferreds
    with pytest.raises(TypeError, match="cannot be used as a Just value"):
        Just(_)
    with pytest.raises(TypeError, match="cannot be used as a Just value"):
        Just(Factory(lambda _: _))


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        {"a": 1, "b": 2},
        {1, 2, 3},
    ],
)
def test_builder_just_unhashable(value):
    p = Just(value)
    assert isinstance(p, JustUnhashable)
    assert p.resolve({}) == value


def test_builder_variable():
    p = Variable("other")
    context = {"other": 10}
    assert p.resolve(context) == 10


def test_builder_factory():
    f = Factory(lambda _: _ + 1)
    assert f.resolve({"_": 1}) == 2
    assert f.resolve({"_": 2}) == 3

    def fn(**kwargs):
        assert kwargs == {"_": 10, "a": 5}
        return -1

    f = Factory(fn)
    assert f.resolve({"_": 10, "a": 5}) == -1


def test_builder_call():
    def fn(a, b, c=1):
        return a + b + c

    c = Call(fn, 1, 2, c=3)
    assert c.resolve({}) == 6

    c = Call(fn, Just(-1), Just(-2))
    assert c.resolve({}) == -2

    c = Call(dict, a=1, b=2)
    assert c.resolve({}) == {"a": 1, "b": 2}

    c = Call(float, "1.1")
    assert c.resolve({}) == 1.1


def test_builder_attr():
    class MyType:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __hash__(self):
            return hash((type(self), self.a, self.b))

    v = Variable("v")
    b = Attr(v, "b")
    assert b.resolve({"v": MyType(1, 2)}) == 2

    b = Attr(MyType(1, 2), "a")
    assert b.resolve({}) == 1

    name = Variable("name")
    # test that name can be a deferred as well
    b = Attr(v, name)
    assert b.resolve({"v": MyType(1, 2), "name": "a"}) == 1


def test_builder_item():
    v = Variable("v")
    b = Item(v, 1)
    assert b.resolve({"v": [1, 2, 3]}) == 2

    b = Item(FrozenDict(a=1, b=2), "a")
    assert b.resolve({}) == 1

    name = Variable("name")
    # test that name can be a deferred as well
    b = Item(v, name)
    assert b.resolve({"v": {"a": 1, "b": 2}, "name": "b"}) == 2


def test_builder_mapping():
    b = Mapping({"a": 1, "b": 2})
    assert b.resolve({}) == {"a": 1, "b": 2}

    b = Mapping({"a": Just(1), "b": Just(2)})
    assert b.resolve({}) == {"a": 1, "b": 2}

    b = Mapping({"a": Just(1), "b": Just(2), "c": _})
    assert b.resolve({"_": 3}) == {"a": 1, "b": 2, "c": 3}


def test_builder():
    class MyClass:
        pass

    def fn(x, ctx):
        return x + 1

    assert resolver(1) == Just(1)
    assert resolver(Just(1)) == Just(1)
    assert resolver(Just(Just(1))) == Just(1)
    assert resolver(MyClass) == Just(MyClass)
    assert resolver(fn) == Factory(fn)
    assert resolver(()) == Sequence(())
    assert resolver((1, 2, _)) == Sequence((Just(1), Just(2), _))
    assert resolver({}) == Mapping({})
    assert resolver({"a": 1, "b": _}) == Mapping({"a": Just(1), "b": _})

    assert resolver(var("x")) == Variable("x")
    assert resolver(Variable("x")) == Variable("x")


def test_builder_objects_are_hashable():
    a = Variable("a")
    b = Attr(a, "b")
    c = Item(a, 1)
    d = Call(operator.add, a, 1)

    set_ = {a, b, c, d}
    assert len(set_) == 4

    for obj in [a, b, c, d]:
        assert obj == obj
        assert hash(obj) == hash(obj)
        set_.add(obj)
        assert len(set_) == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((), ()),
        ([], []),
        ({}, {}),
        ((1, 2, 3), (1, 2, 3)),
        ([1, 2, 3], [1, 2, 3]),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
        (FrozenDict({"a": 1, "b": 2}), FrozenDict({"a": 1, "b": 2})),
    ],
)
def test_deferred_builds(value, expected):
    assert resolver(value).resolve({}) == expected


def test_deferred_supports_string_arguments():
    # deferred() is applied on all arguments of Call() except the first one and
    # sequences are transparently handled, the check far sequences was incorrect
    # for strings causing infinite recursion
    b = resolver("3.14")
    assert b.resolve({}) == "3.14"


def test_deferred_object_are_not_hashable():
    # since __eq__ is overloaded, Deferred objects are not hashable
    with pytest.raises(TypeError, match="unhashable type"):
        hash(_.a)


def test_deferred_const():
    obj = const({"a": 1, "b": 2, "c": "gamma"})

    deferred = obj["c"].upper()
    assert deferred._resolver == Call(Attr(Item(obj, "c"), "upper"))
    assert deferred.resolve() == "GAMMA"


def test_deferred_variable_getattr():
    v = var("v")
    p = v.copy
    assert resolver(p) == Attr(v, "copy")
    assert resolver(p).resolve({"v": [1, 2, 3]})() == [1, 2, 3]

    p = v.copy()
    assert resolver(p) == Call(Attr(v, "copy"))
    assert resolver(p).resolve({"v": [1, 2, 3]}) == [1, 2, 3]


class TableMock(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __eq__(self, other):
        return isinstance(other, TableMock) and super().__eq__(other)


def _binop(name, switch=False):
    def method(self, other):
        if switch:
            return BinaryMock(name=name, left=other, right=self)
        else:
            return BinaryMock(name=name, left=self, right=other)

    return method


class ValueMock(Slotted):
    def log(self, base=None):
        return UnaryMock(name="log", arg=base)

    def sum(self):
        return UnaryMock(name="sum", arg=self)

    def __neg__(self):
        return UnaryMock(name="neg", arg=self)

    def __invert__(self):
        return UnaryMock(name="invert", arg=self)

    __lt__ = _binop("lt")
    __gt__ = _binop("gt")
    __le__ = _binop("le")
    __ge__ = _binop("ge")
    __add__ = _binop("add")
    __radd__ = _binop("add", switch=True)
    __sub__ = _binop("sub")
    __rsub__ = _binop("sub", switch=True)
    __mul__ = _binop("mul")
    __rmul__ = _binop("mul", switch=True)
    __mod__ = _binop("mod")
    __rmod__ = _binop("mod", switch=True)
    __truediv__ = _binop("div")
    __rtruediv__ = _binop("div", switch=True)
    __floordiv__ = _binop("floordiv")
    __rfloordiv__ = _binop("floordiv", switch=True)
    __rshift__ = _binop("shift")
    __rrshift__ = _binop("shift", switch=True)
    __lshift__ = _binop("shift")
    __rlshift__ = _binop("shift", switch=True)
    __pow__ = _binop("pow")
    __rpow__ = _binop("pow", switch=True)
    __xor__ = _binop("xor")
    __rxor__ = _binop("xor", switch=True)
    __and__ = _binop("and")
    __rand__ = _binop("and", switch=True)
    __or__ = _binop("or")
    __ror__ = _binop("or", switch=True)


class ColumnMock(ValueMock):
    __slots__ = ("name", "dtype")

    def __init__(self, name, dtype):
        super().__init__(name=name, dtype=dtype)

    def __deferred_repr__(self):
        return f"<column[{self.dtype}]>"


class UnaryMock(ValueMock):
    __slots__ = ("name", "arg")

    def __init__(self, name, arg):
        super().__init__(name=name, arg=arg)


class BinaryMock(ValueMock):
    __slots__ = ("name", "left", "right")

    def __init__(self, name, left, right):
        super().__init__(name=name, left=left, right=right)


@pytest.fixture
def table():
    return TableMock(
        a=ColumnMock(name="a", dtype="int"),
        b=ColumnMock(name="b", dtype="int"),
        c=ColumnMock(name="c", dtype="string"),
    )


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
def test_deferred_is_pickleable(func, table):
    expr1 = func(_)
    builder1 = resolver(expr1)
    builder2 = pickle.loads(pickle.dumps(builder1))

    r1 = builder1.resolve({"_": table})
    r2 = builder2.resolve({"_": table})

    assert r1 == r2


def test_deferred_getitem(table):
    expr = _["a"]
    assert expr.resolve(table) == table["a"]
    assert repr(expr) == "_['a']"


def test_deferred_getattr(table):
    expr = _.a
    assert expr.resolve(table) == table.a
    assert repr(expr) == "_.a"


def test_deferred_call(table):
    expr = Deferred(Call(operator.add, _.a, 2))
    res = expr.resolve(table)
    assert res == table.a + 2
    assert repr(expr) == "add(_.a, 2)"

    func = lambda a, b: a + b
    expr = Deferred(Call(func, a=_.a, b=2))
    res = expr.resolve(table)
    assert res == table.a + 2
    assert func.__name__ in repr(expr)
    assert "a=_.a, b=2" in repr(expr)

    expr = Deferred(Call(operator.add, (_.a, 2)), repr="<test>")
    assert repr(expr) == "<test>"


def test_deferred_method(table):
    expr = _.a.log()
    res = expr.resolve(table)
    assert res == table.a.log()
    assert repr(expr) == "_.a.log()"


def test_deferred_method_with_args(table):
    expr = _.a.log(1)
    res = expr.resolve(table)
    assert res == table.a.log(1)
    assert repr(expr) == "_.a.log(1)"

    expr = _.a.log(_.b)
    res = expr.resolve(table)
    assert res == table.a.log(table.b)
    assert repr(expr) == "_.a.log(_.b)"


def test_deferred_method_with_kwargs(table):
    expr = _.a.log(base=1)
    res = expr.resolve(table)
    assert res == table.a.log(base=1)
    assert repr(expr) == "_.a.log(base=1)"

    expr = _.a.log(base=_.b)
    res = expr.resolve(table)
    assert res == table.a.log(base=table.b)
    assert repr(expr) == "_.a.log(base=_.b)"


def test_deferred_apply(table):
    expr = Deferred(Call(operator.add, _.a, 2))
    res = expr.resolve(table)
    assert res == table.a + 2
    assert repr(expr) == "add(_.a, 2)"

    func = lambda a, b: a + b
    expr = Deferred(Call(func, _.a, 2))
    res = expr.resolve(table)
    assert res == table.a + 2
    assert func.__name__ in repr(expr)


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
def test_deferred_binary_operations(symbol, op, table):
    expr = op(_.a, _.b)
    sol = op(table.a, table.b)
    res = expr.resolve(table)
    assert res == sol
    assert repr(expr) == f"(_.a {symbol} _.b)"

    expr = op(1, _.a)
    sol = op(1, table.a)
    res = expr.resolve(table)
    assert res == sol
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
def test_deferred_compare_operations(sym, rsym, op, table):
    expr = op(_.a, _.b)
    sol = op(table.a, table.b)
    res = expr.resolve(table)
    assert res == sol
    assert repr(expr) == f"(_.a {sym} _.b)"

    expr = op(1, _.a)
    sol = op(1, table.a)
    res = expr.resolve(table)
    assert res == sol
    assert repr(expr) == f"(_.a {rsym} 1)"


@pytest.mark.parametrize(
    "symbol, op",
    [
        ("-", operator.neg),
        ("~", operator.invert),
    ],
)
def test_deferred_unary_operations(symbol, op, table):
    expr = op(_.a)
    sol = op(table.a)
    res = expr.resolve(table)
    assert res == sol
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

    assert f(table.a, table.b) == table.a + table.b + 3
    assert f(table.a, table.b, c=4) == table.a + table.b + 4

    expr = f(_.a, _.b)
    sol = table.a + table.b + 3
    res = expr.resolve(table)
    assert res == sol
    assert repr(expr) == "f(_.a, _.b)"

    expr = f(1, 2, c=_.a)
    sol = 3 + table.a
    res = expr.resolve(table)
    assert res == sol
    assert repr(expr) == "f(1, 2, c=_.a)"

    with pytest.raises(TypeError, match="unknown"):
        f(_.a, _.b, unknown=3)  # invalid calls caught at call time


def test_deferrable_repr():
    @deferrable(repr="<test>")
    def myfunc(x):
        return x + 1

    assert repr(myfunc(_.a)) == "<test>"


def test_deferred_set_raises():
    with pytest.raises(TypeError, match="unhashable type"):
        {_.a, _.b}  # noqa: B018


@pytest.mark.parametrize(
    "case",
    [
        param(lambda: ([1, _], [1, 2]), id="list"),
        param(lambda: ((1, _), (1, 2)), id="tuple"),
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


def test_deferred_is_final():
    with pytest.raises(TypeError, match="Cannot inherit from final class"):

        class MyDeferred(Deferred):
            pass


def test_deferred_is_immutable():
    with pytest.raises(AttributeError, match="cannot be assigned to immutable"):
        _.a = 1


def test_deferred_namespace(table):
    ns = Namespace(deferred, module=__name__)

    assert isinstance(ns.ColumnMock, Deferred)
    assert resolver(ns.ColumnMock) == Just(ColumnMock)

    d = ns.ColumnMock("a", "int")
    assert resolver(d) == Call(Just(ColumnMock), Just("a"), Just("int"))
    assert d.resolve() == ColumnMock("a", "int")

    d = ns.ColumnMock("a", _)
    assert resolver(d) == Call(Just(ColumnMock), Just("a"), _)
    assert d.resolve("int") == ColumnMock("a", "int")

    a, b = var("a"), var("b")
    d = ns.ColumnMock(a, b).name
    assert d.resolve(a="colname", b="float") == "colname"


def test_custom_deferred_repr(table):
    expr = _.x + table.a
    assert repr(expr) == "(_.x + <column[int]>)"
