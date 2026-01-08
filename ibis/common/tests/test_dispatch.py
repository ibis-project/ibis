from __future__ import annotations

import collections
import decimal

from ibis.common.dispatch import lazy_singledispatch


def test_lazy_singledispatch():
    @lazy_singledispatch
    def foo(_) -> str | int | float | tuple:
        """A docstring."""
        return "base result"

    @foo.register(int)
    def _(x) -> int:
        return x + 1

    @foo.register(float)
    def _(x) -> float:
        return x - 1

    @foo.register((tuple, list))
    def _(x) -> tuple:
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
    def _(a):
        return a + 1

    assert foo(1) == 1
    assert foo(decimal.Decimal(1)) == decimal.Decimal(2)


def test_lazy_singledispatch_lazy_walks_mro():
    class Subclass(decimal.Decimal):
        pass

    class Subclass2(decimal.Decimal):
        pass

    @lazy_singledispatch
    def foo(_):
        return "base call"

    @foo.register(Subclass2)
    def _(_):
        return "eager call"

    @foo.register("decimal.Decimal")
    def _(_):
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
    def foo(_):
        return "base"

    @foo.register(collections.abc.Mapping)
    def _(_):
        return "mapping"

    @foo.register(mydict)
    def _(_):
        return "mydict"

    @foo.register(collections.abc.Callable)
    def _(_):
        return "callable"

    assert foo(1) == "base"
    assert foo({}) == "mapping"
    assert foo(mydict()) == "mydict"  # concrete takes precedence
    assert foo(sum) == "callable"


def test_lazy_singledispatch_finalize():
    @lazy_singledispatch
    def foo(_):
        return "base"

    @foo.register(int)
    def _(_):
        return "int"

    foo.finalize()

    try:

        @foo.register(float)
        def _(_):
            return "float"
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")


def test_lazy_singledispatch_typing():
    # If we declare the original function as returning type T,
    # then all registered implementations must also return type T.
    @lazy_singledispatch
    def returns_int(x) -> int:
        return x

    @returns_int.register(int)
    def _(x) -> int:
        return x + 1

    @returns_int.register(str)  # ty:ignore[invalid-argument-type]
    def _(x) -> str:
        return x + "!"

    # And all the results should be typed as int

    def func_taking_int(x: int):
        return x

    def func_taking_str(x: str):
        return x

    inty = returns_int(1)
    func_taking_int(inty)
    func_taking_str(inty)  # ty:ignore[invalid-argument-type]
