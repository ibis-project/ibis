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


def _to_sqlglot(arg):
    return arg if isinstance(arg, sg.exp.Expression) else lit(arg)


def _func(name: str, *args: Any, **kwargs: Any):
    return sg.func(name, *map(_to_sqlglot, args), **kwargs)


class FuncGen:
    __slots__ = ()

    def __getattr__(self, name: str) -> partial:
        return partial(_func, name)

    def __getitem__(self, key: str) -> partial:
        return getattr(self, key)

    def array(self, *args):
        return sg.exp.Array.from_arg_list(list(map(_to_sqlglot, args)))

    def tuple(self, *args):
        return sg.func("tuple", *map(_to_sqlglot, args))

    def exists(self, query):
        return sg.exp.Exists(this=query)

    def concat(self, *args):
        return sg.exp.Concat.from_arg_list(list(map(_to_sqlglot, args)))


def lit(val):
    return sg.exp.Literal(this=str(val), is_string=isinstance(val, str))


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
