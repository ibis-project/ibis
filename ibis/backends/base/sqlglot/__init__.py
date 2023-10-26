from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import sqlglot as sg

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    from ibis.backends.base.sqlglot.datatypes import SqlglotType


class AggGen:
    __slots__ = ("aggfunc",)

    def __init__(self, *, aggfunc: Callable) -> None:
        self.aggfunc = aggfunc

    def __getattr__(self, name: str) -> partial:
        return partial(self.aggfunc, name)

    def __getitem__(self, key: str) -> partial:
        return getattr(self, key)


def _func(name: str, *args: Any, **kwargs: Any):
    return sg.func(name, *map(sg.exp.convert, args), **kwargs)


class FuncGen:
    __slots__ = ()

    def __getattr__(self, name: str) -> partial:
        return partial(_func, name)

    def __getitem__(self, key: str) -> partial:
        return getattr(self, key)

    def array(self, *args):
        return sg.exp.Array.from_arg_list(list(map(sg.exp.convert, args)))

    def tuple(self, *args):
        return sg.func("tuple", *map(sg.exp.convert, args))

    def exists(self, query):
        return sg.exp.Exists(this=query)

    def concat(self, *args):
        return sg.exp.Concat.from_arg_list(list(map(sg.exp.convert, args)))

    def map(self, keys, values):
        return sg.exp.Map(keys=keys, values=values)


class ColGen:
    __slots__ = ()

    def __getattr__(self, name: str) -> sg.exp.Column:
        return sg.column(name)

    def __getitem__(self, key: str) -> sg.exp.Column:
        return sg.column(key)


def paren(expr):
    """Wrap a sqlglot expression in parentheses."""
    return sg.exp.Paren(this=expr)


def parenthesize(op, arg):
    import ibis.expr.operations as ops

    if isinstance(op, (ops.Binary, ops.Unary)):
        return paren(arg)
    # function calls don't need parens
    return arg


def interval(value, *, unit):
    return sg.exp.Interval(this=sg.exp.convert(value), unit=sg.exp.var(unit))


C = ColGen()
F = FuncGen()
NULL = sg.exp.NULL
FALSE = sg.exp.FALSE
TRUE = sg.exp.TRUE
STAR = sg.exp.Star()


def make_cast(
    converter: SqlglotType,
) -> Callable[[sg.exp.Expression, dt.DataType], sg.exp.Cast]:
    def cast(arg: sg.exp.Expression, to: dt.DataType) -> sg.exp.Cast:
        return sg.cast(arg, to=converter.from_ibis(to))

    return cast
