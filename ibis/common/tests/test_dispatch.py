from __future__ import annotations

import collections
import decimal
from typing import TYPE_CHECKING, Union

from ibis.common.dispatch import Dispatched, lazy_singledispatch

# ruff: noqa: F811
if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


def test_lazy_singledispatch():
    @lazy_singledispatch
    def foo(x):
        """A docstring."""
        return "base result"

    @foo.register(int)
    def _(x):
        return x + 1

    @foo.register(float)
    def _(x):
        return x - 1

    @foo.register((tuple, list))
    def _(x):
        return tuple(foo(i) for i in x)

    class Bar:
        pass

    assert foo.__name__ == "foo"
    assert foo.__doc__ == "A docstring."

    assert foo(1) == 2
    assert foo(1.0) == 0.0
    assert foo((1, 2.0)) == (2, 1.0)
    assert foo([1, 2.0]) == (2, 1.0)
    assert foo("a string") == "base result"


def test_lazy_singledispatch_extra_args():
    @lazy_singledispatch
    def foo(*args, **kwargs):
        pass

    @foo.register(int)
    def _(a, b, c=2):
        return a + b + c

    @foo.register(float)
    def _(a, b, c=2):
        return a - b - c

    assert foo(1, 2) == 5
    assert foo(1.0, 2) == -3.0
    assert foo(1, 2, c=3) == 6


def test_lazy_singledispatch_lazy():
    @lazy_singledispatch
    def foo(a):
        return a

    @foo.register("decimal.Decimal")
    def inc(a):
        return a + 1

    assert foo(1) == 1
    assert foo(decimal.Decimal(1)) == decimal.Decimal(2)


def test_lazy_singledispatch_lazy_walks_mro():
    class Subclass(decimal.Decimal):
        pass

    class Subclass2(decimal.Decimal):
        pass

    @lazy_singledispatch
    def foo(a):
        return "base call"

    @foo.register(Subclass2)
    def _(a):
        return "eager call"

    @foo.register("decimal.Decimal")
    def _(a):
        return "lazy call"

    assert foo(1) == "base call"
    assert foo(Subclass2(1)) == "eager call"
    assert foo(Subclass(1)) == "lazy call"
    # not overwritten by lazy loader
    assert foo(Subclass2(1)) == "eager call"


def test_lazy_singledispatch_abc():
    class mydict(dict):
        pass

    @lazy_singledispatch
    def foo(a):
        return "base"

    @foo.register(collections.abc.Mapping)
    def _(a):
        return "mapping"

    @foo.register(mydict)
    def _(a):
        return "mydict"

    @foo.register(collections.abc.Callable)
    def _(a):
        return "callable"

    assert foo(1) == "base"
    assert foo({}) == "mapping"
    assert foo(mydict()) == "mydict"  # concrete takes precedence
    assert foo(sum) == "callable"


class A:
    pass


class B:
    pass


class Visitor(Dispatched):
    def a(self):
        return "a"

    def b(self, x: int):
        return "b_int"

    def b(self, x: str):
        return "b_str"

    def b(self, x: Union[A, B]):
        return "b_union"

    @classmethod
    def c(cls, x: int, **kwargs):
        return "c_int"

    @classmethod
    def c(cls, x: str, a=0, b=1):
        return "c_str"

    def d(self, x: int):
        return "d_int"

    def d(self, x: str):
        return "d_str"

    @staticmethod
    def e(x: int):
        return "e_int"

    @staticmethod
    def e(x: str):
        return "e_str"

    def f(self, df: dict):
        return "f_dict"

    def f(self, df: pd.DataFrame):
        return "f_pandas"

    def f(self, df: pa.Table):
        return "f_pyarrow"


class Subvisitor(Visitor):
    def b(self, x):
        return super().b(x)

    def b(self, x: float):
        return "b_float"

    @classmethod
    def c(cls, x):
        return super().c(x)

    @classmethod
    def c(cls, s: float):
        return "c_float"


def test_dispatched():
    v = Visitor()
    assert v.a() == "a"
    assert v.b(1) == "b_int"
    assert v.b("1") == "b_str"
    assert v.b(A()) == "b_union"
    assert v.b(B()) == "b_union"
    assert v.d(1) == "d_int"
    assert v.d("1") == "d_str"

    w = Subvisitor()
    assert w.b(1) == "b_int"
    assert w.b(1.1) == "b_float"

    assert Visitor.c(1, a=0, b=0) == "c_int"
    assert Visitor.c("1") == "c_str"

    assert Visitor.e("1") == "e_str"
    assert Visitor.e(1) == "e_int"

    assert Subvisitor.c(1) == "c_int"
    assert Subvisitor.c(1.1) == "c_float"

    assert Subvisitor.e(1) == "e_int"


def test_dispatched_lazy():
    import pyarrow as pa

    empty_pyarrow_table = pa.Table.from_arrays([])
    empty_pandas_table = empty_pyarrow_table.to_pandas()

    v = Visitor()
    assert v.f({}) == "f_dict"
    assert v.f(empty_pyarrow_table) == "f_pyarrow"
    assert v.f(empty_pandas_table) == "f_pandas"
