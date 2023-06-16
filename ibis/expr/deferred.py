from __future__ import annotations

import operator
from typing import Any, Callable

_BINARY_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "**": operator.pow,
    "%": operator.mod,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    ">>": operator.rshift,
    "<<": operator.lshift,
}

_UNARY_OPS: dict[str, Callable[[Any], Any]] = {
    "~": operator.inv,
    "-": operator.neg,
}


class Deferred:
    """A deferred expression."""

    __slots__ = ()

    def resolve(self, param: Any) -> Any:
        """Resolve the deferred expression against an argument.

        Parameters
        ----------
        param
            The argument to use in place of `_` in the expression.

        Returns
        -------
        Any
            The result of resolving the deferred expression.
        """
        return self._resolve(param)

    def __repr__(self) -> str:
        return "_"

    def __hash__(self) -> int:
        return id(self)

    def _resolve(self, param: Any) -> Any:
        return param

    def __getattr__(self, attr: str) -> Deferred:
        if attr.startswith("__"):
            raise AttributeError(f"'Deferred' object has no attribute {attr!r}")
        return DeferredAttr(self, attr)

    def __getitem__(self, key: Any) -> Deferred:
        return DeferredItem(self, key)

    def __call__(self, *args: Any, **kwargs: Any) -> Deferred:
        return DeferredCall(self, *args, **kwargs)

    def __add__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("+", self, other)

    def __radd__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("+", other, self)

    def __sub__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("-", self, other)

    def __rsub__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("-", other, self)

    def __mul__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("*", self, other)

    def __rmul__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("*", other, self)

    def __truediv__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("/", self, other)

    def __rtruediv__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("/", other, self)

    def __floordiv__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("//", self, other)

    def __rfloordiv__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("//", other, self)

    def __pow__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("**", self, other)

    def __rpow__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("**", other, self)

    def __mod__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("%", self, other)

    def __rmod__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("%", other, self)

    def __rshift__(self, other: Any) -> Deferred:
        return DeferredBinaryOp(">>", self, other)

    def __rrshift__(self, other: Any) -> Deferred:
        return DeferredBinaryOp(">>", other, self)

    def __lshift__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("<<", self, other)

    def __rlshift__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("<<", other, self)

    def __eq__(self, other: Any) -> Deferred:  # type: ignore
        return DeferredBinaryOp("==", self, other)

    def __ne__(self, other: Any) -> Deferred:  # type: ignore
        return DeferredBinaryOp("!=", self, other)

    def __lt__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("<", self, other)

    def __le__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("<=", self, other)

    def __gt__(self, other: Any) -> Deferred:
        return DeferredBinaryOp(">", self, other)

    def __ge__(self, other: Any) -> Deferred:
        return DeferredBinaryOp(">=", self, other)

    def __and__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("&", self, other)

    def __rand__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("&", other, self)

    def __or__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("|", self, other)

    def __ror__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("|", other, self)

    def __xor__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("^", self, other)

    def __rxor__(self, other: Any) -> Deferred:
        return DeferredBinaryOp("^", other, self)

    def __invert__(self) -> Deferred:
        return DeferredUnaryOp("~", self)

    def __neg__(self) -> Deferred:
        return DeferredUnaryOp("-", self)


class DeferredAttr(Deferred):
    __slots__ = ("_value", "_attr")

    def __init__(self, value: Any, attr: str) -> None:
        self._value = value
        self._attr = attr

    def __repr__(self) -> str:
        return f"{self._value!r}.{self._attr}"

    def _resolve(self, param: Any) -> Any:
        obj = _resolve(self._value, param)
        return getattr(obj, self._attr)


class DeferredItem(Deferred):
    __slots__ = ("_value", "_key")

    def __init__(self, value: Any, key: Any) -> None:
        self._value = value
        self._key = key

    def __repr__(self) -> str:
        return f"{self._value!r}[{self._key!r}]"

    def _resolve(self, param: Any) -> Any:
        obj = _resolve(self._value, param)
        return obj[self._key]


class DeferredCall(Deferred):
    __slots__ = ("_func", "_args", "_kwargs")

    def __init__(self, func: Any, *args: Any, **kwargs: Any) -> None:
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        params = [repr(a) for a in self._args]
        params.extend(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"{self._func!r}({', '.join(params)})"

    def _resolve(self, param: Any) -> Any:
        func = _resolve(self._func, param)
        args = [_resolve(a, param) for a in self._args]
        kwargs = {k: _resolve(v, param) for k, v in self._kwargs.items()}
        return func(*args, **kwargs)


class DeferredBinaryOp(Deferred):
    __slots__ = ("_symbol", "_left", "_right")

    def __init__(self, symbol: str, left: Any, right: Any) -> None:
        self._symbol = symbol
        self._left = left
        self._right = right

    def __repr__(self) -> str:
        return f"({self._left!r} {self._symbol} {self._right!r})"

    def _resolve(self, param: Any) -> Any:
        left = _resolve(self._left, param)
        right = _resolve(self._right, param)
        return _BINARY_OPS[self._symbol](left, right)


class DeferredUnaryOp(Deferred):
    __slots__ = ("_symbol", "_value")

    def __init__(self, symbol: str, value: Any) -> None:
        self._symbol = symbol
        self._value = value

    def __repr__(self) -> str:
        return f"{self._symbol}{self._value!r}"

    def _resolve(self, param: Any) -> Any:
        value = _resolve(self._value, param)
        return _UNARY_OPS[self._symbol](value)


def _resolve(expr: Deferred, param: Any) -> Any:
    if isinstance(expr, Deferred):
        return expr._resolve(param)
    return expr


def deferred_apply(func: Callable, *args: Any, **kwargs: Any) -> Deferred:
    """Construct a deferred call from a callable and arguments.

    Parameters
    ----------
    func
        The callable to defer
    args
        Positional arguments, possibly composed of deferred expressions.
    kwargs
        Keyword arguments, possible composed of deferred expressions.

    Returns
    -------
    expr
        A deferred expression representing the call.
    """
    return DeferredCall(func, *args, **kwargs)
