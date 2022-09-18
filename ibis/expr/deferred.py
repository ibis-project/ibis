from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable

import toolz

if TYPE_CHECKING:
    import ibis.expr.types as ir


class Deferred:
    """A deferred expression."""

    __slots__ = ("resolve",)

    def __init__(self, resolve: Callable = toolz.identity) -> None:
        assert callable(
            resolve
        ), f"resolve argument is not callable, got {type(resolve)}"
        self.resolve = resolve

    def __hash__(self) -> int:
        # every new instance can potentially be referring to something
        # different so treat each instance as unique
        return id(self)

    def _defer(self, func: Callable, *args: Any, **kwargs: Any) -> Deferred:
        """Wrap `func` in a `Deferred` instance."""

        def resolve(expr, func=func, self=self, args=args, kwargs=kwargs):
            resolved_expr = self.resolve(expr)
            resolved_args = [_resolve(arg, expr=expr) for arg in args]
            resolved_kwargs = {
                name: _resolve(arg, expr=expr) for name, arg in kwargs.items()
            }
            return func(resolved_expr, *resolved_args, **resolved_kwargs)

        return self.__class__(resolve)

    def __getattr__(self, name: str) -> Deferred:
        return self._defer(getattr, name)

    def __getitem__(self, key: Any) -> Deferred:
        return self._defer(operator.itemgetter(key))

    def __call__(self, *args: Any, **kwargs: Any) -> Deferred:
        return self._defer(
            lambda expr, *args, **kwargs: expr(*args, **kwargs),
            *args,
            **kwargs,
        )

    def __add__(self, other: Any) -> Deferred:
        return self._defer(operator.add, other)

    def __radd__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.add), other)

    def __sub__(self, other: Any) -> Deferred:
        return self._defer(operator.sub, other)

    def __rsub__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.sub), other)

    def __mul__(self, other: Any) -> Deferred:
        return self._defer(operator.mul, other)

    def __rmul__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.mul), other)

    def __truediv__(self, other: Any) -> Deferred:
        return self._defer(operator.truediv, other)

    def __rtruediv__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.truediv), other)

    def __floordiv__(self, other: Any) -> Deferred:
        return self._defer(operator.floordiv, other)

    def __rfloordiv__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.floordiv), other)

    def __pow__(self, other: Any) -> Deferred:
        return self._defer(operator.pow, other)

    def __rpow__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.pow), other)

    def __mod__(self, other: Any) -> Deferred:
        return self._defer(operator.mod, other)

    def __rmod__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.mod), other)

    def __eq__(self, other: Any) -> Deferred:  # type: ignore
        return self._defer(operator.eq, other)

    def __ne__(self, other: Any) -> Deferred:  # type: ignore
        return self._defer(operator.ne, other)

    def __lt__(self, other: Any) -> Deferred:
        return self._defer(operator.lt, other)

    def __le__(self, other: Any) -> Deferred:
        return self._defer(operator.le, other)

    def __gt__(self, other: Any) -> Deferred:
        return self._defer(operator.gt, other)

    def __ge__(self, other: Any) -> Deferred:
        return self._defer(operator.ge, other)

    def __or__(self, other: Any) -> Deferred:
        return self._defer(operator.or_, other)

    def __ror__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.or_), other)

    def __and__(self, other: Any) -> Deferred:
        return self._defer(operator.and_, other)

    def __rand__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.and_), other)

    def __xor__(self, other: Any) -> Deferred:
        return self._defer(operator.xor, other)

    def __rxor__(self, other: Any) -> Deferred:
        return self._defer(toolz.flip(operator.xor), other)

    def __invert__(self) -> Deferred:
        return self._defer(operator.invert)

    def __neg__(self) -> Deferred:
        return self._defer(operator.neg)


def _resolve(arg: Any, *, expr: ir.Expr) -> Any:
    try:
        return arg.resolve(expr)
    except AttributeError:
        return arg
