from __future__ import annotations

import collections.abc
import functools
import inspect
import operator
from abc import abstractmethod
from typing import Any, Callable, TypeVar, overload

from ibis.common.bases import Final, FrozenSlotted, Hashable, Immutable, Slotted
from ibis.common.collections import FrozenDict
from ibis.common.typing import Coercible, CoercionError
from ibis.util import PseudoHashable, is_iterable


class Resolver(Coercible, Hashable):
    """Specification about constructing a value given a context.

    The context is a dictionary that contains all the captured values and
    information relevant for the builder.

    The builder is used in the right hand side of the replace pattern:
    `Replace(pattern, builder)`. Semantically when a match occurs for the
    replace pattern, the builder is called with the context and the result
    of the builder is used as the replacement value.
    """

    @abstractmethod
    def resolve(self, context: dict):
        """Construct a new object from the context.

        Parameters
        ----------
        context
            A dictionary containing all the captured values and information
            relevant for the deferred.

        Returns
        -------
        The constructed object.
        """

    @abstractmethod
    def __eq__(self, other: Resolver) -> bool:
        ...

    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, Deferred):
            return value._resolver
        else:
            raise CoercionError(f"Cannot coerce {value!r} to {cls.__name__!r}")


class Deferred(Slotted, Immutable, Final):
    """The user facing wrapper object providing syntactic sugar for deferreds.

    Provides a natural-like syntax for constructing deferred expressions by
    overloading all of the available dunder methods including the equality
    operator.

    Its sole purpose is to provide a nicer syntax for constructing deferred
    expressions, thus it gets unwrapped to the underlying deferred expression
    when used by the rest of the library.

    Parameters
    ----------
    deferred
        The deferred object to provide syntax sugar for.
    repr
        An optional fixed string to use when repr-ing the deferred expression,
        instead of the default. This is useful for complex deferred expressions
        where the arguments don't necessarily make sense to be user facing in
        the repr.
    """

    __slots__ = ("_resolver", "_repr")

    def __init__(self, obj, repr=None):
        super().__init__(_resolver=resolver(obj), _repr=repr)

    # TODO(kszucs): consider to make this method protected
    def resolve(self, _=None, **kwargs):
        context = {"_": _, **kwargs}
        return self._resolver.resolve(context)

    def __repr__(self):
        return repr(self._resolver) if self._repr is None else self._repr

    def __getattr__(self, name):
        return Deferred(Attr(self, name))

    def __iter__(self):
        raise TypeError(f"{self.__class__.__name__!r} object is not iterable")

    def __getitem__(self, name):
        return Deferred(Item(self, name))

    def __call__(self, *args, **kwargs):
        return Deferred(Call(self, *args, **kwargs))

    def __invert__(self) -> Deferred:
        return Deferred(UnaryOperator(operator.invert, self))

    def __neg__(self) -> Deferred:
        return Deferred(UnaryOperator(operator.neg, self))

    def __add__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.add, self, other))

    def __radd__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.add, other, self))

    def __sub__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.sub, self, other))

    def __rsub__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.sub, other, self))

    def __mul__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.mul, self, other))

    def __rmul__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.mul, other, self))

    def __truediv__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.truediv, self, other))

    def __rtruediv__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.truediv, other, self))

    def __floordiv__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.floordiv, self, other))

    def __rfloordiv__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.floordiv, other, self))

    def __pow__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.pow, self, other))

    def __rpow__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.pow, other, self))

    def __mod__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.mod, self, other))

    def __rmod__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.mod, other, self))

    def __rshift__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.rshift, self, other))

    def __rrshift__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.rshift, other, self))

    def __lshift__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.lshift, self, other))

    def __rlshift__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.lshift, other, self))

    def __eq__(self, other: Any) -> Deferred:  # type: ignore
        return Deferred(BinaryOperator(operator.eq, self, other))

    def __ne__(self, other: Any) -> Deferred:  # type: ignore
        return Deferred(BinaryOperator(operator.ne, self, other))

    def __lt__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.lt, self, other))

    def __le__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.le, self, other))

    def __gt__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.gt, self, other))

    def __ge__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.ge, self, other))

    def __and__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.and_, self, other))

    def __rand__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.and_, other, self))

    def __or__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.or_, self, other))

    def __ror__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.or_, other, self))

    def __xor__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.xor, self, other))

    def __rxor__(self, other: Any) -> Deferred:
        return Deferred(BinaryOperator(operator.xor, other, self))


class Variable(FrozenSlotted, Resolver):
    """Retrieve a value from the context.

    Parameters
    ----------
    name
        The key to retrieve from the state.
    """

    __slots__ = ("name",)
    name: Any

    def __init__(self, name):
        super().__init__(name=name)

    def __repr__(self):
        return str(self.name)

    def resolve(self, context):
        return context[self.name]


class Just(FrozenSlotted, Resolver):
    """Construct exactly the given value.

    Parameters
    ----------
    value
        The value to return when the deferred is called.
    """

    __slots__ = ("value",)
    value: Any

    @classmethod
    def __create__(cls, value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, (Deferred, Resolver)):
            raise TypeError(f"{value} cannot be used as a Just value")
        elif isinstance(value, collections.abc.Hashable):
            return super().__create__(value)
        else:
            return JustUnhashable(value)

    def __init__(self, value):
        super().__init__(value=value)

    def __repr__(self):
        obj = self.value
        if hasattr(obj, "__deferred_repr__"):
            return obj.__deferred_repr__()
        elif callable(obj):
            return getattr(obj, "__name__", repr(obj))
        else:
            return repr(obj)

    def resolve(self, context):
        return self.value


class JustUnhashable(FrozenSlotted, Resolver):
    """Construct exactly the given unhashable value.

    Parameters
    ----------
    value
        The value to return when the deferred is called.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        hashable_value = PseudoHashable(value)
        super().__init__(value=hashable_value)

    def __repr__(self):
        obj = self.value.obj
        if hasattr(obj, "__deferred_repr__"):
            return obj.__deferred_repr__()
        elif callable(obj):
            return getattr(obj, "__name__", repr(obj))
        else:
            return repr(obj)

    def resolve(self, context):
        return self.value.obj


class Factory(FrozenSlotted, Resolver):
    """Construct a value by calling a function.

    The function is called with two positional arguments:
    1. the value being matched
    2. the context dictionary

    The function must return the constructed value.

    Parameters
    ----------
    func
        The function to apply.
    """

    __slots__ = ("func",)
    func: Callable

    def __init__(self, func):
        assert callable(func)
        super().__init__(func=func)

    def resolve(self, context):
        return self.func(**context)


class Attr(FrozenSlotted, Resolver):
    __slots__ = ("obj", "name")
    obj: Resolver
    name: str

    def __init__(self, obj, name):
        super().__init__(obj=resolver(obj), name=resolver(name))

    def __repr__(self):
        if isinstance(self.name, Just):
            return f"{self.obj!r}.{self.name.value}"
        else:
            return f"Attr({self.obj!r}, {self.name!r})"

    def resolve(self, context):
        obj = self.obj.resolve(context)
        name = self.name.resolve(context)
        return getattr(obj, name)


class Item(FrozenSlotted, Resolver):
    __slots__ = ("obj", "name")
    obj: Resolver
    name: str

    def __init__(self, obj, name):
        super().__init__(obj=resolver(obj), name=resolver(name))

    def __repr__(self):
        if isinstance(self.name, Just):
            return f"{self.obj!r}[{self.name.value!r}]"
        else:
            return f"Item({self.obj!r}, {self.name!r})"

    def resolve(self, context):
        obj = self.obj.resolve(context)
        name = self.name.resolve(context)
        return obj[name]


class Call(FrozenSlotted, Resolver):
    """Pattern that calls a function with the given arguments.

    Both positional and keyword arguments are coerced into patterns.

    Parameters
    ----------
    func
        The function to call.
    args
        The positional argument patterns.
    kwargs
        The keyword argument patterns.
    """

    __slots__ = ("func", "args", "kwargs")
    func: Resolver
    args: tuple[Resolver, ...]
    kwargs: dict[str, Resolver]

    def __init__(self, func, *args, **kwargs):
        if isinstance(func, Deferred):
            func = func._resolver
        elif isinstance(func, Resolver):
            pass
        elif callable(func):
            func = Just(func)
        else:
            raise TypeError(f"Invalid callable {func!r}")
        args = tuple(map(resolver, args))
        kwargs = FrozenDict({k: resolver(v) for k, v in kwargs.items()})
        super().__init__(func=func, args=args, kwargs=kwargs)

    def resolve(self, context):
        func = self.func.resolve(context)
        args = tuple(arg.resolve(context) for arg in self.args)
        kwargs = {k: v.resolve(context) for k, v in self.kwargs.items()}
        return func(*args, **kwargs)

    def __repr__(self):
        func = repr(self.func)
        args = ", ".join(map(repr, self.args))
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        if args and kwargs:
            return f"{func}({args}, {kwargs})"
        elif args:
            return f"{func}({args})"
        elif kwargs:
            return f"{func}({kwargs})"
        else:
            return f"{func}()"


_operator_symbols = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    operator.floordiv: "//",
    operator.pow: "**",
    operator.mod: "%",
    operator.eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    operator.and_: "&",
    operator.or_: "|",
    operator.xor: "^",
    operator.rshift: ">>",
    operator.lshift: "<<",
    operator.inv: "~",
    operator.neg: "-",
    operator.invert: "~",
}


class UnaryOperator(FrozenSlotted, Resolver):
    __slots__ = ("func", "arg")
    func: Callable
    arg: Resolver

    def __init__(self, func, arg):
        assert func in _operator_symbols
        super().__init__(func=func, arg=resolver(arg))

    def __repr__(self):
        symbol = _operator_symbols[self.func]
        return f"{symbol}{self.arg!r}"

    def resolve(self, context):
        arg = self.arg.resolve(context)
        return self.func(arg)


class BinaryOperator(FrozenSlotted, Resolver):
    __slots__ = ("func", "left", "right")
    func: Callable
    left: Resolver
    right: Resolver

    def __init__(self, func, left, right):
        assert func in _operator_symbols
        super().__init__(func=func, left=resolver(left), right=resolver(right))

    def __repr__(self):
        symbol = _operator_symbols[self.func]
        return f"({self.left!r} {symbol} {self.right!r})"

    def resolve(self, context):
        left = self.left.resolve(context)
        right = self.right.resolve(context)
        return self.func(left, right)


class Mapping(FrozenSlotted, Resolver):
    __slots__ = ("typ", "values")

    def __init__(self, values):
        typ = type(values)
        values = FrozenDict({k: resolver(v) for k, v in values.items()})
        super().__init__(typ=typ, values=values)

    def __repr__(self):
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.values.items())
        if self.typ is dict:
            return f"{{{items}}}"
        else:
            return f"{self.typ.__name__}({{{items}}})"

    def resolve(self, context):
        items = {k: v.resolve(context) for k, v in self.values.items()}
        return self.typ(items)


class Sequence(FrozenSlotted, Resolver):
    __slots__ = ("typ", "values")
    typ: type

    def __init__(self, values):
        typ = type(values)
        values = tuple(map(resolver, values))
        super().__init__(typ=typ, values=values)

    def __repr__(self):
        elems = ", ".join(map(repr, self.values))
        if self.typ is tuple:
            return f"({elems})"
        elif self.typ is list:
            return f"[{elems}]"
        else:
            return f"{self.typ.__name__}({elems})"

    def resolve(self, context):
        return self.typ(v.resolve(context) for v in self.values)


def resolver(obj):
    if isinstance(obj, Deferred):
        return obj._resolver
    elif isinstance(obj, Resolver):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        # allow nesting deferred patterns in dicts
        return Mapping(obj)
    elif is_iterable(obj):
        # allow nesting deferred patterns in tuples/lists
        return Sequence(obj)
    elif isinstance(obj, type):
        return Just(obj)
    elif callable(obj):
        return Factory(obj)
    else:
        # the object is used as a constant value
        return Just(obj)


def deferred(obj):
    return Deferred(resolver(obj))


def var(name):
    return Deferred(Variable(name))


def const(value):
    return Deferred(Just(value))


def _contains_deferred(obj: Any) -> bool:
    if isinstance(obj, (Resolver, Deferred)):
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
                builder = Call(func, *args, **kwargs)
                return Deferred(builder, repr=repr)
            return func(*args, **kwargs)

        return inner  # type: ignore

    return wrapper if func is None else wrapper(func)


# reserved variable name for the value being matched
_ = var("_")
