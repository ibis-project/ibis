from __future__ import annotations

import functools
import inspect
import operator
from typing import Any, Callable, NoReturn, TypeVar, overload

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


def _repr(x: Any) -> str:
    """A helper for nicely repring deferred expressions."""
    import ibis.expr.types as ir

    if isinstance(x, ir.Column):
        return f"<column[{x.type()}]>"
    elif isinstance(x, ir.Scalar):
        return f"<scalar[{x.type()}]>"
    return repr(x)


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

    def __iter__(self) -> NoReturn:
        raise TypeError(f"{self.__class__.__name__!r} object is not iterable")

    def __getattr__(self, attr: str) -> Deferred:
        if attr.startswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {attr!r}"
            )
        return DeferredAttr(self, attr)

    def __getitem__(self, key: Any) -> Deferred:
        return DeferredItem(self, key)

    def __call__(self, *args: Any, **kwargs: Any) -> Deferred:
        return DeferredCall(self, args, kwargs)

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
    _value: Any
    _attr: str

    def __init__(self, value: Any, attr: str) -> None:
        self._value = value
        self._attr = attr

    def __repr__(self) -> str:
        return f"{_repr(self._value)}.{self._attr}"

    def _resolve(self, param: Any) -> Any:
        obj = _resolve(self._value, param)
        return getattr(obj, self._attr)


class DeferredItem(Deferred):
    __slots__ = ("_value", "_key")
    _value: Any
    _key: Any

    def __init__(self, value: Any, key: Any) -> None:
        self._value = value
        self._key = key

    def __repr__(self) -> str:
        return f"{_repr(self._value)}[{_repr(self._key)}]"

    def _resolve(self, param: Any) -> Any:
        obj = _resolve(self._value, param)
        return obj[self._key]


class DeferredCall(Deferred):
    __slots__ = ("_func", "_args", "_kwargs", "_repr")
    _func: Callable
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]
    _repr: str | None

    def __init__(
        self,
        func: Any,
        args: tuple | None = None,
        kwargs: dict | None = None,
        repr: str | None = None,
    ) -> None:
        self._func = func
        self._args = args or ()
        self._kwargs = kwargs or {}
        self._repr = repr

    def __repr__(self) -> str:
        if self._repr is not None:
            return self._repr

        params = [_repr(a) for a in self._args]
        params.extend(f"{k}={_repr(v)}" for k, v in self._kwargs.items())
        # Repr directly wrapped functions as their name, fallback to repr for
        # deferred objects or callables without __name__ otherwise
        func = getattr(self._func, "__name__", "") or repr(self._func)
        return f"{func}({', '.join(params)})"

    def _resolve(self, param: Any) -> Any:
        func = _resolve(self._func, param)
        args = [_resolve(a, param) for a in self._args]
        kwargs = {k: _resolve(v, param) for k, v in self._kwargs.items()}
        return func(*args, **kwargs)


class DeferredBinaryOp(Deferred):
    __slots__ = ("_symbol", "_left", "_right")
    _symbol: str
    _left: Any
    _right: Any

    def __init__(self, symbol: str, left: Any, right: Any) -> None:
        self._symbol = symbol
        self._left = left
        self._right = right

    def __repr__(self) -> str:
        return f"({_repr(self._left)} {self._symbol} {_repr(self._right)})"

    def _resolve(self, param: Any) -> Any:
        left = _resolve(self._left, param)
        right = _resolve(self._right, param)
        return _BINARY_OPS[self._symbol](left, right)


class DeferredUnaryOp(Deferred):
    __slots__ = ("_symbol", "_value")
    _symbol: str
    _value: Any

    def __init__(self, symbol: str, value: Any) -> None:
        self._symbol = symbol
        self._value = value

    def __repr__(self) -> str:
        return f"{self._symbol}{_repr(self._value)}"

    def _resolve(self, param: Any) -> Any:
        value = _resolve(self._value, param)
        return _UNARY_OPS[self._symbol](value)


def _resolve(expr: Any, param: Any) -> Any:
    if isinstance(expr, Deferred):
        return expr._resolve(param)
    elif (typ := type(expr)) in (tuple, list, set):
        return typ(_resolve(e, param) for e in expr)
    elif typ is dict:
        return {k: _resolve(v, param) for k, v in expr.items()}
    else:
        return expr


def _contains_deferred(obj: Any) -> bool:
    if isinstance(obj, Deferred):
        return True
    elif (typ := type(obj)) in (tuple, list, set):
        return any(_contains_deferred(o) for o in obj)
    elif typ is dict:
        return any(_contains_deferred(o) for o in obj.values())
    return False


F = TypeVar("F", bound=Callable)


@overload
def deferrable(*, repr: str | None = None) -> Callable[[F], F]:
    ...


@overload
def deferrable(func: F) -> F:
    ...


def deferrable(func=None, *, repr=None):
    """Wrap a top-level expr function to support deferred arguments.

    When a deferrable function is called, the args & kwargs are traversed to
    look for `Deferred` values (through builtin collections like
    `list`/`tuple`/`set`/`dict`). If any `Deferred` arguments are found, then
    the result is also `Deferred`. Otherwise the function is called directly.

    Parameters
    ----------
    func
        A callable to make deferrable
    repr
        An optional fixed string to use when repr-ing the deferred expression,
        instead of the usual. This is useful for complex deferred expressions
        where the arguments don't necessarily make sense to be user facing
        in the repr.
    """

    def wrapper(func):
        # Parse the signature of func so we can validate deferred calls eagerly,
        # erroring for invalid/missing arguments at call time not resolve time.
        sig = inspect.signature(func)

        @functools.wraps(func)
        def inner(*args, **kwargs):
            if _contains_deferred((args, kwargs)):
                # Try to bind the arguments now, raising a nice error
                # immediately if the function was called incorrectly
                sig.bind(*args, **kwargs)
                return DeferredCall(func, args, kwargs, repr=repr)
            return func(*args, **kwargs)

        return inner  # type: ignore

    return wrapper if func is None else wrapper(func)
