from __future__ import annotations

import math
import numbers
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from inspect import Parameter
from itertools import chain
from typing import (
    Annotated,
    ForwardRef,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)
from typing import Any as AnyType

import toolz
from typing_extensions import GenericMeta, get_args, get_origin

from ibis.common.bases import FrozenSlotted as Slotted
from ibis.common.bases import Hashable, Singleton
from ibis.common.collections import FrozenDict, RewindableIterator, frozendict
from ibis.common.deferred import (
    Deferred,
    Resolver,
    Variable,
    _,  # noqa: F401
    resolver,
)
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.typing import (
    Coercible,
    CoercionError,
    Sentinel,
    UnionType,
    _ClassInfo,
    format_typehint,
    get_bound_typevars,
    get_type_params,
)
from ibis.util import is_iterable, promote_tuple

T_co = TypeVar("T_co", covariant=True)


class NoMatch(metaclass=Sentinel):
    """Marker to indicate that a pattern didn't match."""


# TODO(kszucs): have an As[int] or Coerced[int] type in ibis.common.typing which
# would be used to annotate an argument as coercible to int or to a certain type
# without needing for the type to inherit from Coercible
class Pattern(Hashable):
    """Base class for all patterns.

    Patterns are used to match values against a given condition. They are extensively
    used by other core components of Ibis to validate and/or coerce user inputs.
    """

    @classmethod
    def from_typehint(cls, annot: type, allow_coercion: bool = True) -> Pattern:
        """Construct a validator from a python type annotation.

        Parameters
        ----------
        annot
            The typehint annotation to construct the pattern from. This must be
            an already evaluated type annotation.
        allow_coercion
            Whether to use coercion if the typehint is a Coercible type.

        Returns
        -------
        A pattern that matches the given type annotation.
        """
        # TODO(kszucs): cache the result of this function
        # TODO(kszucs): explore issubclass(typ, SupportsInt) etc.
        origin, args = get_origin(annot), get_args(annot)

        if origin is None:
            # the typehint is not generic
            if annot is Ellipsis or annot is AnyType:
                # treat both `Any` and `...` as wildcard
                return _any
            elif isinstance(annot, type):
                # the typehint is a concrete type (e.g. int, str, etc.)
                if allow_coercion and issubclass(annot, Coercible):
                    # the type implements the Coercible protocol so we try to
                    # coerce the value to the given type rather than checking
                    return CoercedTo(annot)
                else:
                    return InstanceOf(annot)
            elif isinstance(annot, TypeVar):
                # if the typehint is a type variable we try to construct a
                # validator from it only if it is covariant and has a bound
                if not annot.__covariant__:
                    raise NotImplementedError(
                        "Only covariant typevars are supported for now"
                    )
                if annot.__bound__:
                    return cls.from_typehint(annot.__bound__)
                else:
                    return _any
            elif isinstance(annot, Enum):
                # for enums we check the value against the enum values
                return EqualTo(annot)
            elif isinstance(annot, str):
                # for strings and forward references we check in a lazy way
                return LazyInstanceOf(annot)
            elif isinstance(annot, ForwardRef):
                return LazyInstanceOf(annot.__forward_arg__)
            else:
                raise TypeError(f"Cannot create validator from annotation {annot!r}")
        elif origin is CoercedTo:
            return CoercedTo(args[0])
        elif origin is Literal:
            # for literal types we check the value against the literal values
            return IsIn(args)
        elif origin is UnionType or origin is Union:
            # this is slightly more complicated because we need to handle
            # Optional[T] which is Union[T, None] and Union[T1, T2, ...]
            *rest, last = args
            if last is type(None):
                # the typehint is Optional[*rest] which is equivalent to
                # Union[*rest, None], so we construct an Option pattern
                if len(rest) == 1:
                    inner = cls.from_typehint(rest[0])
                else:
                    inner = AnyOf(*map(cls.from_typehint, rest))
                return Option(inner)
            else:
                # the typehint is Union[*args] so we construct an AnyOf pattern
                return AnyOf(*map(cls.from_typehint, args))
        elif origin is Annotated:
            # the Annotated typehint can be used to add extra validation logic
            # to the typehint, e.g. Annotated[int, Positive], the first argument
            # is used for isinstance checks, the rest are applied in conjunction
            annot, *extras = args
            return AllOf(cls.from_typehint(annot), *extras)
        elif origin is Callable:
            # the Callable typehint is used to annotate functions, e.g. the
            # following typehint annotates a function that takes two integers
            # and returns a string: Callable[[int, int], str]
            if args:
                # callable with args and return typehints construct a special
                # CallableWith validator
                arg_hints, return_hint = args
                arg_patterns = tuple(map(cls.from_typehint, arg_hints))
                return_pattern = cls.from_typehint(return_hint)
                return CallableWith(arg_patterns, return_pattern)
            else:
                # in case of Callable without args we check for the Callable
                # protocol only
                return InstanceOf(Callable)
        elif issubclass(origin, tuple):
            # construct validators for the tuple elements, but need to treat
            # variadic tuples differently, e.g. tuple[int, ...] is a variadic
            # tuple of integers, while tuple[int] is a tuple with a single int
            first, *rest = args
            # TODO(kszucs): consider to support the same SequenceOf path if args
            # has a single element, e.g. tuple[int] since annotation a single
            # element tuple is not common OR use typing.Sequence for annotating
            # instead of tuple[T, ...] OR have a VarTupleOf pattern
            if rest == [Ellipsis]:
                inners = cls.from_typehint(first)
            else:
                inners = tuple(map(cls.from_typehint, args))
            return TupleOf(inners)
        elif issubclass(origin, Sequence):
            # construct a validator for the sequence elements where all elements
            # must be of the same type, e.g. Sequence[int] is a sequence of ints
            (value_inner,) = map(cls.from_typehint, args)
            return SequenceOf(value_inner, type=origin)
        elif issubclass(origin, Mapping):
            # construct a validator for the mapping keys and values, e.g.
            # Mapping[str, int] is a mapping with string keys and int values
            key_inner, value_inner = map(cls.from_typehint, args)
            return MappingOf(key_inner, value_inner, type=origin)
        elif isinstance(origin, GenericMeta):
            # construct a validator for the generic type, see the specific
            # Generic* validators for more details
            if allow_coercion and issubclass(origin, Coercible) and args:
                return GenericCoercedTo(annot)
            else:
                return GenericInstanceOf(annot)
        else:
            raise TypeError(
                f"Cannot create validator from annotation {annot!r} {origin!r}"
            )

    @abstractmethod
    def match(self, value: AnyType, context: dict[str, AnyType]) -> AnyType:
        """Match a value against the pattern.

        Parameters
        ----------
        value
            The value to match the pattern against.
        context
            A dictionary providing arbitrary context for the pattern matching.

        Returns
        -------
        The result of the pattern matching. If the pattern doesn't match
        the value, then it must return the `NoMatch` sentinel value.
        """
        ...

    def describe(self, plural=False):
        return f"matching {self!r}"

    @abstractmethod
    def __eq__(self, other: Pattern) -> bool:
        ...

    def __invert__(self) -> Not:
        """Syntax sugar for matching the inverse of the pattern."""
        return Not(self)

    def __or__(self, other: Pattern) -> AnyOf:
        """Syntax sugar for matching either of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if either of the patterns match.
        """
        return AnyOf(self, other)

    def __and__(self, other: Pattern) -> AllOf:
        """Syntax sugar for matching both of the patterns.

        Parameters
        ----------
        other
            The other pattern to match against.

        Returns
        -------
        New pattern that matches if both of the patterns match.
        """
        return AllOf(self, other)

    def __rshift__(self, other: Deferred) -> Replace:
        """Syntax sugar for replacing a value.

        Parameters
        ----------
        other
            The deferred to use for constructing the replacement value.

        Returns
        -------
        New replace pattern.
        """
        return Replace(self, other)

    def __rmatmul__(self, name: str) -> Capture:
        """Syntax sugar for capturing a value.

        Parameters
        ----------
        name
            The name of the capture.

        Returns
        -------
        New capture pattern.
        """
        return Capture(name, self)


class Is(Slotted, Pattern):
    """Pattern that matches a value against a reference value.

    Parameters
    ----------
    value
        The reference value to match against.
    """

    __slots__ = ("value",)
    value: AnyType

    def match(self, value, context):
        if value is self.value:
            return value
        else:
            return NoMatch


class Any(Slotted, Singleton, Pattern):
    """Pattern that accepts any value, basically a no-op."""

    def match(self, value, context):
        return value


_any = Any()


class Nothing(Slotted, Singleton, Pattern):
    """Pattern that no values."""

    def match(self, value, context):
        return NoMatch


class Capture(Slotted, Pattern):
    """Pattern that captures a value in the context.

    Parameters
    ----------
    pattern
        The pattern to match against.
    key
        The key to use in the context if the pattern matches.
    """

    __slots__ = ("key", "pattern")
    key: AnyType
    pattern: Pattern

    def __init__(self, key, pat=_any):
        if isinstance(key, (Deferred, Resolver)):
            key = resolver(key)
            if isinstance(key, Variable):
                key = key.name
            else:
                raise TypeError("Only variables can be used as capture keys")
        super().__init__(key=key, pattern=pattern(pat))

    def match(self, value, context):
        value = self.pattern.match(value, context)
        if value is NoMatch:
            return NoMatch
        context[self.key] = value
        return value


class Replace(Slotted, Pattern):
    """Pattern that replaces a value with the output of another pattern.

    Parameters
    ----------
    matcher
        The pattern to match against.
    resolver
        The deferred to use as a replacement.
    """

    __slots__ = ("pattern", "resolver")
    pattern: Pattern
    resolver: Resolver

    def __init__(self, matcher, replacer):
        super().__init__(pattern=pattern(matcher), resolver=resolver(replacer))

    def match(self, value, context):
        value = self.pattern.match(value, context)
        if value is NoMatch:
            return NoMatch
        # use the `_` reserved variable to record the value being replaced
        # in the context, so that it can be used in the replacer pattern
        context["_"] = value
        return self.resolver.resolve(context)


def replace(matcher):
    """More convenient syntax for replacing a value with the output of a function."""

    def decorator(replacer):
        return Replace(matcher, replacer)

    return decorator


class Check(Slotted, Pattern):
    """Pattern that checks a value against a predicate.

    Parameters
    ----------
    predicate
        The predicate to use.
    """

    __slots__ = ("predicate",)
    predicate: Callable

    @classmethod
    def __create__(cls, predicate):
        if isinstance(predicate, (Deferred, Resolver)):
            return DeferredCheck(predicate)
        else:
            return super().__create__(predicate)

    def __init__(self, predicate):
        assert callable(predicate)
        super().__init__(predicate=predicate)

    def describe(self, plural=False):
        if plural:
            return f"values that satisfy {self.predicate.__name__}()"
        else:
            return f"a value that satisfies {self.predicate.__name__}()"

    def match(self, value, context):
        if self.predicate(value):
            return value
        else:
            return NoMatch


If = Check


class DeferredCheck(Slotted, Pattern):
    __slots__ = ("resolver",)
    resolver: Resolver

    def __init__(self, obj):
        super().__init__(resolver=resolver(obj))

    def describe(self, plural=False):
        if plural:
            return f"values that satisfy {self.resolver!r}"
        else:
            return f"a value that satisfies {self.resolver!r}"

    def match(self, value, context):
        context["_"] = value
        if self.resolver.resolve(context):
            return value
        else:
            return NoMatch


class Custom(Slotted, Pattern):
    """User defined custom matcher function.

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

    def match(self, value, context):
        return self.func(value, context)


class EqualTo(Slotted, Pattern):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.
    """

    __slots__ = ("value",)
    value: AnyType

    @classmethod
    def __create__(cls, value):
        if isinstance(value, (Deferred, Resolver)):
            return DeferredEqualTo(value)
        else:
            return super().__create__(value)

    def __init__(self, value):
        super().__init__(value=value)

    def match(self, value, context):
        if value == self.value:
            return value
        else:
            return NoMatch

    def describe(self, plural=False):
        return repr(self.value)


Eq = EqualTo


class DeferredEqualTo(Slotted, Pattern):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.
    """

    __slots__ = ("resolver",)
    resolver: Resolver

    def __init__(self, obj):
        super().__init__(resolver=resolver(obj))

    def match(self, value, context):
        context["_"] = value
        if value == self.resolver.resolve(context):
            return value
        else:
            return NoMatch

    def describe(self, plural=False):
        return repr(self.resolver)


class Option(Slotted, Pattern):
    """Pattern that matches `None` or a value that passes the inner validator.

    Parameters
    ----------
    pattern
        The inner pattern to use.
    """

    __slots__ = ("pattern", "default")
    pattern: Pattern
    default: AnyType

    def __init__(self, pat, default=None):
        super().__init__(pattern=pattern(pat), default=default)

    def describe(self, plural=False):
        if plural:
            return f"optional {self.pattern.describe(plural=True)}"
        else:
            return f"either None or {self.pattern.describe(plural=False)}"

    def match(self, value, context):
        if value is None:
            if self.default is None:
                return None
            else:
                return self.default
        else:
            return self.pattern.match(value, context)


def _describe_type(typ, plural=False):
    if isinstance(typ, tuple):
        *rest, last = typ
        rest = ", ".join(_describe_type(t, plural=plural) for t in rest)
        last = _describe_type(last, plural=plural)
        return f"{rest} or {last}" if rest else last

    name = format_typehint(typ)
    if plural:
        return f"{name}s"
    elif name[0].lower() in "aeiou":
        return f"an {name}"
    else:
        return f"a {name}"


class TypeOf(Slotted, Pattern):
    """Pattern that matches a value that is of a given type."""

    __slots__ = ("type",)
    type: type

    def __init__(self, typ):
        super().__init__(type=typ)

    def describe(self, plural=False):
        return f"exactly {_describe_type(self.type, plural=plural)}"

    def match(self, value, context):
        if type(value) is self.type:
            return value
        else:
            return NoMatch


class SubclassOf(Slotted, Pattern):
    """Pattern that matches a value that is a subclass of a given type.

    Parameters
    ----------
    type
        The type to check against.
    """

    __slots__ = ("type",)

    def __init__(self, typ):
        super().__init__(type=typ)

    def describe(self, plural=False):
        if plural:
            return f"subclasses of {self.type.__name__}"
        else:
            return f"a subclass of {self.type.__name__}"

    def match(self, value, context):
        if issubclass(value, self.type):
            return value
        else:
            return NoMatch


class InstanceOf(Slotted, Singleton, Pattern):
    """Pattern that matches a value that is an instance of a given type.

    Parameters
    ----------
    types
        The type to check against.
    """

    __slots__ = ("type",)
    type: _ClassInfo

    def __init__(self, typ):
        super().__init__(type=typ)

    def describe(self, plural=False):
        return _describe_type(self.type, plural=plural)

    def match(self, value, context):
        if isinstance(value, self.type):
            return value
        else:
            return NoMatch

    def __call__(self, *args, **kwargs):
        return Object(self.type, *args, **kwargs)


class GenericInstanceOf(Slotted, Pattern):
    """Pattern that matches a value that is an instance of a given generic type.

    Parameters
    ----------
    typ
        The type to check against (must be a generic type).

    Examples
    --------
    >>> class MyNumber(Generic[T_co]):
    ...     value: T_co
    ...
    ...     def __init__(self, value: T_co):
    ...         self.value = value
    ...
    ...     def __eq__(self, other):
    ...         return type(self) is type(other) and self.value == other.value
    ...
    >>> p = GenericInstanceOf(MyNumber[int])
    >>> assert p.match(MyNumber(1), {}) == MyNumber(1)
    >>> assert p.match(MyNumber(1.0), {}) is NoMatch
    >>>
    >>> p = GenericInstanceOf(MyNumber[float])
    >>> assert p.match(MyNumber(1.0), {}) == MyNumber(1.0)
    >>> assert p.match(MyNumber(1), {}) is NoMatch
    """

    __slots__ = ("type", "origin", "fields")
    origin: type
    fields: FrozenDict[str, Pattern]

    def __init__(self, typ):
        origin = get_origin(typ)
        typevars = get_bound_typevars(typ)

        fields = {}
        for var, (attr, type_) in typevars.items():
            if not var.__covariant__:
                raise TypeError(
                    f"Typevar {var} is not covariant, cannot use it in a GenericInstanceOf"
                )
            fields[attr] = Pattern.from_typehint(type_, allow_coercion=False)

        super().__init__(type=typ, origin=origin, fields=frozendict(fields))

    def describe(self, plural=False):
        return _describe_type(self.type, plural=plural)

    def match(self, value, context):
        if not isinstance(value, self.origin):
            return NoMatch

        for name, pattern in self.fields.items():
            attr = getattr(value, name)
            if pattern.match(attr, context) is NoMatch:
                return NoMatch

        return value


class LazyInstanceOf(Slotted, Pattern):
    """A version of `InstanceOf` that accepts qualnames instead of imported classes.

    Useful for delaying imports.

    Parameters
    ----------
    types
        The types to check against.
    """

    __slots__ = ("types", "check")
    types: tuple[type, ...]
    check: Callable

    def __init__(self, types):
        types = promote_tuple(types)
        check = lazy_singledispatch(lambda x: False)
        check.register(types, lambda x: True)
        super().__init__(types=types, check=check)

    def match(self, value, context):
        if self.check(value):
            return value
        else:
            return NoMatch


class CoercedTo(Slotted, Pattern, Generic[T_co]):
    """Force a value to have a particular Python type.

    If a Coercible subclass is passed, the `__coerce__` method will be used to
    coerce the value. Otherwise, the type will be called with the value as the
    only argument.

    Parameters
    ----------
    type
        The type to coerce to.
    """

    __slots__ = ("type", "func")
    type: T_co

    def __init__(self, type):
        func = type.__coerce__ if issubclass(type, Coercible) else type
        super().__init__(type=type, func=func)

    def describe(self, plural=False):
        type = _describe_type(self.type, plural=False)
        if plural:
            return f"coercibles to {type}"
        else:
            return f"coercible to {type}"

    def match(self, value, context):
        try:
            value = self.func(value)
        except (TypeError, CoercionError):
            return NoMatch

        if isinstance(value, self.type):
            return value
        else:
            return NoMatch

    def __call__(self, *args, **kwargs):
        return Object(self.type, *args, **kwargs)


As = CoercedTo


class GenericCoercedTo(Slotted, Pattern):
    """Force a value to have a particular generic Python type.

    Parameters
    ----------
    typ
        The type to coerce to. Must be a generic type with bound typevars.

    Examples
    --------
    >>> from typing import Generic, TypeVar
    >>>
    >>> T = TypeVar("T", covariant=True)
    >>>
    >>> class MyNumber(Coercible, Generic[T]):
    ...     __slots__ = ("value",)
    ...
    ...     def __init__(self, value):
    ...         self.value = value
    ...
    ...     def __eq__(self, other):
    ...         return type(self) is type(other) and self.value == other.value
    ...
    ...     @classmethod
    ...     def __coerce__(cls, value, T=None):
    ...         if issubclass(T, int):
    ...             return cls(int(value))
    ...         elif issubclass(T, float):
    ...             return cls(float(value))
    ...         else:
    ...             raise CoercionError(f"Cannot coerce to {T}")
    ...
    >>> p = GenericCoercedTo(MyNumber[int])
    >>> assert p.match(3.14, {}) == MyNumber(3)
    >>> assert p.match("15", {}) == MyNumber(15)
    >>>
    >>> p = GenericCoercedTo(MyNumber[float])
    >>> assert p.match(3.14, {}) == MyNumber(3.14)
    >>> assert p.match("15", {}) == MyNumber(15.0)
    """

    __slots__ = ("origin", "params", "checker")
    origin: type
    params: FrozenDict[str, type]
    checker: GenericInstanceOf

    def __init__(self, target):
        origin = get_origin(target)
        checker = GenericInstanceOf(target)
        params = frozendict(get_type_params(target))
        super().__init__(origin=origin, params=params, checker=checker)

    def describe(self, plural=False):
        if plural:
            return f"coercibles to {self.checker.describe(plural=False)}"
        else:
            return f"coercible to {self.checker.describe(plural=False)}"

    def match(self, value, context):
        try:
            value = self.origin.__coerce__(value, **self.params)
        except CoercionError:
            return NoMatch

        if self.checker.match(value, context) is NoMatch:
            return NoMatch

        return value


class Not(Slotted, Pattern):
    """Pattern that matches a value that does not match a given pattern.

    Parameters
    ----------
    pattern
        The pattern which the value should not match.
    """

    __slots__ = ("pattern",)
    pattern: Pattern

    def __init__(self, inner):
        super().__init__(pattern=pattern(inner))

    def describe(self, plural=False):
        if plural:
            return f"anything except {self.pattern.describe(plural=True)}"
        else:
            return f"anything except {self.pattern.describe(plural=False)}"

    def match(self, value, context):
        if self.pattern.match(value, context) is NoMatch:
            return value
        else:
            return NoMatch


class AnyOf(Slotted, Pattern):
    """Pattern that if any of the given patterns match.

    Parameters
    ----------
    patterns
        The patterns to match against. The first pattern that matches will be
        returned.
    """

    __slots__ = ("patterns",)
    patterns: tuple[Pattern, ...]

    def __init__(self, *pats):
        patterns = tuple(map(pattern, pats))
        super().__init__(patterns=patterns)

    def describe(self, plural=False):
        *rest, last = self.patterns
        rest = ", ".join(p.describe(plural=plural) for p in rest)
        last = last.describe(plural=plural)
        return f"{rest} or {last}" if rest else last

    def match(self, value, context):
        for pattern in self.patterns:
            result = pattern.match(value, context)
            if result is not NoMatch:
                return result
        return NoMatch


class AllOf(Slotted, Pattern):
    """Pattern that matches if all of the given patterns match.

    Parameters
    ----------
    patterns
        The patterns to match against. The value will be passed through each
        pattern in order. The changes applied to the value propagate through the
        patterns.
    """

    __slots__ = ("patterns",)
    patterns: tuple[Pattern, ...]

    def __init__(self, *pats):
        patterns = tuple(map(pattern, pats))
        super().__init__(patterns=patterns)

    def describe(self, plural=False):
        *rest, last = self.patterns
        rest = ", ".join(p.describe(plural=plural) for p in rest)
        last = last.describe(plural=plural)
        return f"{rest} then {last}" if rest else last

    def match(self, value, context):
        for pattern in self.patterns:
            value = pattern.match(value, context)
            if value is NoMatch:
                return NoMatch
        return value


class Length(Slotted, Pattern):
    """Pattern that matches if the length of a value is within a given range.

    Parameters
    ----------
    exactly
        The exact length of the value. If specified, `at_least` and `at_most`
        must be None.
    at_least
        The minimum length of the value.
    at_most
        The maximum length of the value.
    """

    __slots__ = ("at_least", "at_most")
    at_least: int
    at_most: int

    def __init__(
        self,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        if exactly is not None:
            if at_least is not None or at_most is not None:
                raise ValueError("Can't specify both exactly and at_least/at_most")
            at_least = exactly
            at_most = exactly
        super().__init__(at_least=at_least, at_most=at_most)

    def describe(self, plural=False):
        if self.at_least is not None and self.at_most is not None:
            if self.at_least == self.at_most:
                return f"with length exactly {self.at_least}"
            else:
                return f"with length between {self.at_least} and {self.at_most}"
        elif self.at_least is not None:
            return f"with length at least {self.at_least}"
        elif self.at_most is not None:
            return f"with length at most {self.at_most}"
        else:
            return "with any length"

    def match(self, value, context):
        length = len(value)
        if self.at_least is not None and length < self.at_least:
            return NoMatch
        if self.at_most is not None and length > self.at_most:
            return NoMatch
        return value


class Between(Slotted, Pattern):
    """Match a value between two bounds.

    Parameters
    ----------
    lower
        The lower bound.
    upper
        The upper bound.
    """

    __slots__ = ("lower", "upper")
    lower: float
    upper: float

    def __init__(self, lower: float = -math.inf, upper: float = math.inf):
        super().__init__(lower=lower, upper=upper)

    def match(self, value, context):
        if self.lower <= value <= self.upper:
            return value
        else:
            return NoMatch


class Contains(Slotted, Pattern):
    """Pattern that matches if a value contains a given value.

    Parameters
    ----------
    needle
        The item that the passed value should contain.
    """

    __slots__ = ("needle",)
    needle: AnyType

    def __init__(self, needle):
        super().__init__(needle=needle)

    def describe(self, plural=False):
        return f"containing {self.needle!r}"

    def match(self, value, context):
        if self.needle in value:
            return value
        else:
            return NoMatch


class IsIn(Slotted, Pattern):
    """Pattern that matches if a value is in a given set.

    Parameters
    ----------
    haystack
        The set of values that the passed value should be in.
    """

    __slots__ = ("haystack",)
    haystack: frozenset

    def __init__(self, haystack):
        super().__init__(haystack=frozenset(haystack))

    def describe(self, plural=False):
        return f"in {set(self.haystack)!r}"

    def match(self, value, context):
        if value in self.haystack:
            return value
        else:
            return NoMatch


In = IsIn


class SequenceOf(Slotted, Pattern):
    """Pattern that matches if all of the items in a sequence match a given pattern.

    Specialization of the more flexible GenericSequenceOf pattern which uses two
    additional patterns to possibly coerce the sequence type and to match on
    the length of the sequence.

    Parameters
    ----------
    item
        The pattern to match against each item in the sequence.
    type
        The type to coerce the sequence to. Defaults to tuple.
    """

    __slots__ = ("item", "type")
    item: Pattern
    type: type

    @classmethod
    def __create__(
        cls,
        item,
        type: type = tuple,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        if (
            exactly is not None
            or at_least is not None
            or at_most is not None
            or issubclass(type, Coercible)
        ):
            return GenericSequenceOf(
                item, type=type, exactly=exactly, at_least=at_least, at_most=at_most
            )
        return super().__create__(item, type=type)

    def __init__(self, item, type=tuple):
        super().__init__(item=pattern(item), type=type)

    def describe(self, plural=False):
        typ = _describe_type(self.type, plural=plural)
        item = self.item.describe(plural=True)
        return f"{typ} of {item}"

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        result = []
        for item in values:
            item = self.item.match(item, context)
            if item is NoMatch:
                return NoMatch
            result.append(item)

        return self.type(result)


class GenericSequenceOf(Slotted, Pattern):
    """Pattern that matches if all of the items in a sequence match a given pattern.

    Parameters
    ----------
    item
        The pattern to match against each item in the sequence.
    type
        The type to coerce the sequence to. Defaults to tuple.
    exactly
        The exact length of the sequence.
    at_least
        The minimum length of the sequence.
    at_most
        The maximum length of the sequence.
    """

    __slots__ = ("item", "type", "length")
    item: Pattern
    type: Pattern
    length: Length

    @classmethod
    def __create__(
        cls,
        item: Pattern,
        type: type = tuple,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        if (
            exactly is None
            and at_least is None
            and at_most is None
            and not issubclass(type, Coercible)
        ):
            return SequenceOf(item, type=type)
        else:
            return super().__create__(item, type, exactly, at_least, at_most)

    def __init__(
        self,
        item: Pattern,
        type: type = tuple,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        item = pattern(item)
        type = CoercedTo(type)
        length = Length(at_least=at_least, at_most=at_most)
        super().__init__(item=item, type=type, length=length)

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        result = []
        for value in values:
            value = self.item.match(value, context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        result = self.type.match(result, context)
        if result is NoMatch:
            return NoMatch

        return self.length.match(result, context)


class TupleOf(Slotted, Pattern):
    """Pattern that matches if the respective items in a tuple match the given patterns.

    Parameters
    ----------
    fields
        The patterns to match the respective items in the tuple.
    """

    __slots__ = ("fields",)
    fields: tuple[Pattern, ...]

    @classmethod
    def __create__(cls, fields):
        if not isinstance(fields, tuple):
            return SequenceOf(fields, tuple)
        return super().__create__(fields)

    def __init__(self, fields):
        fields = tuple(map(pattern, fields))
        super().__init__(fields=fields)

    def describe(self, plural=False):
        fields = ", ".join(f.describe(plural=False) for f in self.fields)
        if plural:
            return f"tuples of ({fields})"
        else:
            return f"a tuple of ({fields})"

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        if len(values) != len(self.fields):
            return NoMatch

        result = []
        for pattern, value in zip(self.fields, values):
            value = pattern.match(value, context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        return tuple(result)


class GenericMappingOf(Slotted, Pattern):
    """Pattern that matches if all of the keys and values match the given patterns.

    Parameters
    ----------
    key
        The pattern to match the keys against.
    value
        The pattern to match the values against.
    type
        The type to coerce the mapping to. Defaults to dict.
    """

    __slots__ = ("key", "value", "type")
    key: Pattern
    value: Pattern
    type: Pattern

    def __init__(self, key: Pattern, value: Pattern, type: type = dict):
        super().__init__(key=pattern(key), value=pattern(value), type=CoercedTo(type))

    def match(self, value, context):
        if not isinstance(value, Mapping):
            return NoMatch

        result = {}
        for k, v in value.items():
            if (k := self.key.match(k, context)) is NoMatch:
                return NoMatch
            if (v := self.value.match(v, context)) is NoMatch:
                return NoMatch
            result[k] = v

        result = self.type.match(result, context)
        if result is NoMatch:
            return NoMatch

        return result


MappingOf = GenericMappingOf


class Attrs(Slotted, Pattern):
    __slots__ = ("fields",)
    fields: FrozenDict[str, Pattern]

    def __init__(self, **fields):
        fields = frozendict(toolz.valmap(pattern, fields))
        super().__init__(fields=fields)

    def match(self, value, context):
        for attr, pattern in self.fields.items():
            if not hasattr(value, attr):
                return NoMatch

            v = getattr(value, attr)
            if match(pattern, v, context) is NoMatch:
                return NoMatch

        return value


class Object(Slotted, Pattern):
    """Pattern that matches if the object has the given attributes and they match the given patterns.

    The type must conform the structural pattern matching protocol, e.g. it must have a
    __match_args__ attribute that is a tuple of the names of the attributes to match.

    Parameters
    ----------
    type
        The type of the object.
    *args
        The positional arguments to match against the attributes of the object.
    **kwargs
        The keyword arguments to match against the attributes of the object.
    """

    __slots__ = ("type", "args", "kwargs")
    type: Pattern
    args: tuple[Pattern, ...]
    kwargs: FrozenDict[str, Pattern]

    @classmethod
    def __create__(cls, type, *args, **kwargs):
        if not args and not kwargs:
            return InstanceOf(type)
        return super().__create__(type, *args, **kwargs)

    def __init__(self, type, *args, **kwargs):
        type = pattern(type)
        args = tuple(map(pattern, args))
        kwargs = frozendict(toolz.valmap(pattern, kwargs))
        super().__init__(type=type, args=args, kwargs=kwargs)

    def match(self, value, context):
        if self.type.match(value, context) is NoMatch:
            return NoMatch

        patterns = {**self.kwargs, **dict(zip(value.__match_args__, self.args))}

        fields = {}
        changed = False
        for name, pattern in patterns.items():
            try:
                attr = getattr(value, name)
            except AttributeError:
                return NoMatch

            result = pattern.match(attr, context)
            if result is NoMatch:
                return NoMatch
            elif result != attr:
                changed = True
                fields[name] = result
            else:
                fields[name] = attr

        if changed:
            return type(value)(**fields)
        else:
            return value


class Node(Slotted, Pattern):
    __slots__ = ("type", "each_arg")
    type: Pattern

    def __init__(self, type, each_arg):
        super().__init__(type=pattern(type), each_arg=pattern(each_arg))

    def match(self, value, context):
        if self.type.match(value, context) is NoMatch:
            return NoMatch

        newargs = {}
        changed = False
        for name, arg in zip(value.__argnames__, value.__args__):
            result = self.each_arg.match(arg, context)
            if result is NoMatch:
                newargs[name] = arg
            else:
                newargs[name] = result
                changed = True

        if changed:
            return value.__class__(**newargs)
        else:
            return value


class CallableWith(Slotted, Pattern):
    __slots__ = ("args", "return_")
    args: tuple
    return_: AnyType

    def __init__(self, args, return_=_any):
        super().__init__(args=tuple(args), return_=return_)

    def match(self, value, context):
        from ibis.common.annotations import EMPTY, annotated

        if not callable(value):
            return NoMatch

        fn = annotated(self.args, self.return_, value)

        has_varargs = False
        positional, required_positional = [], []
        for p in fn.__signature__.parameters.values():
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(p)
                if p.default is EMPTY:
                    required_positional.append(p)
            elif p.kind is Parameter.KEYWORD_ONLY and p.default is EMPTY:
                raise TypeError(
                    "Callable has mandatory keyword-only arguments which cannot be specified"
                )
            elif p.kind is Parameter.VAR_POSITIONAL:
                has_varargs = True

        if len(required_positional) > len(self.args):
            # Callable has more positional arguments than expected")
            return NoMatch
        elif len(positional) < len(self.args) and not has_varargs:
            # Callable has less positional arguments than expected")
            return NoMatch
        else:
            return fn


class PatternSequence(Slotted, Pattern):
    # TODO(kszucs): add a length optimization to not even try to match if the
    # length of the sequence is lower than the length of the pattern sequence

    __slots__ = ("pattern_window",)
    pattern_window: tuple[tuple[Pattern, Pattern], ...]

    def __init__(self, patterns):
        current_patterns = [
            SequenceOf(_any) if p is Ellipsis else pattern(p) for p in patterns
        ]
        following_patterns = chain(current_patterns[1:], [Not(_any)])
        pattern_window = tuple(zip(current_patterns, following_patterns))
        super().__init__(pattern_window=pattern_window)

    def match(self, value, context):
        it = RewindableIterator(value)
        result = []

        if not self.pattern_window:
            try:
                next(it)
            except StopIteration:
                return result
            else:
                return NoMatch

        for current, following in self.pattern_window:
            original = current

            if isinstance(current, Capture):
                current = current.pattern
            if isinstance(following, Capture):
                following = following.pattern

            if isinstance(current, (SequenceOf, GenericSequenceOf, PatternSequence)):
                if isinstance(following, (SequenceOf, GenericSequenceOf)):
                    following = following.item
                elif isinstance(following, PatternSequence):
                    # first pattern to match from the pattern window
                    following = following.pattern_window[0][0]

                matches = []
                while True:
                    it.checkpoint()
                    try:
                        item = next(it)
                    except StopIteration:
                        break

                    res = following.match(item, context)
                    if res is NoMatch:
                        matches.append(item)
                    else:
                        it.rewind()
                        break

                res = original.match(matches, context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.extend(res)
            else:
                try:
                    item = next(it)
                except StopIteration:
                    return NoMatch

                res = original.match(item, context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.append(res)

        return result


class PatternMapping(Slotted, Pattern):
    __slots__ = ("keys", "values")
    keys: PatternSequence
    values: PatternSequence

    def __init__(self, patterns):
        keys = PatternSequence(list(map(pattern, patterns.keys())))
        values = PatternSequence(list(map(pattern, patterns.values())))
        super().__init__(keys=keys, values=values)

    def match(self, value, context):
        if not isinstance(value, Mapping):
            return NoMatch

        keys = value.keys()
        if (keys := self.keys.match(keys, context)) is NoMatch:
            return NoMatch

        values = value.values()
        if (values := self.values.match(values, context)) is NoMatch:
            return NoMatch

        return dict(zip(keys, values))


def NoneOf(*args) -> Pattern:
    """Match none of the passed patterns."""
    return Not(AnyOf(*args))


def ListOf(pattern):
    """Match a list of items matching the given pattern."""
    return SequenceOf(pattern, type=list)


def DictOf(key_pattern, value_pattern):
    """Match a dictionary with keys and values matching the given patterns."""
    return MappingOf(key_pattern, value_pattern, type=dict)


def FrozenDictOf(key_pattern, value_pattern):
    """Match a frozendict with keys and values matching the given patterns."""
    return MappingOf(key_pattern, value_pattern, type=frozendict)


def pattern(obj: AnyType) -> Pattern:
    """Create a pattern from various types.

    Not that if a Coercible type is passed as argument, the constructed pattern
    won't attempt to coerce the value during matching. In order to allow type
    coercions use `Pattern.from_typehint()` factory method.

    Parameters
    ----------
    obj
        The object to create a pattern from. Can be a pattern, a type, a callable,
        a mapping, an iterable or a value.

    Examples
    --------
    >>> assert pattern(Any()) == Any()
    >>> assert pattern(int) == InstanceOf(int)
    >>>
    >>> @pattern
    ... def as_int(x, context):
    ...     return int(x)
    ...
    >>>
    >>> assert as_int.match(1, {}) == 1

    Returns
    -------
    The constructed pattern.
    """
    if obj is Ellipsis:
        return _any
    elif isinstance(obj, Pattern):
        return obj
    elif isinstance(obj, (Deferred, Resolver)):
        return Capture(obj)
    elif isinstance(obj, Mapping):
        return PatternMapping(obj)
    elif isinstance(obj, type):
        return InstanceOf(obj)
    elif get_origin(obj):
        return Pattern.from_typehint(obj, allow_coercion=False)
    elif is_iterable(obj):
        return PatternSequence(obj)
    elif callable(obj):
        return Custom(obj)
    else:
        return EqualTo(obj)


def match(
    pat: Pattern, value: AnyType, context: Optional[dict[str, AnyType]] = None
) -> Any:
    """Match a value against a pattern.

    Parameters
    ----------
    pat
        The pattern to match against.
    value
        The value to match.
    context
        Arbitrary mapping of values to be used while matching.

    Returns
    -------
    The matched value if the pattern matches, otherwise :obj:`NoMatch`.

    Examples
    --------
    >>> assert match(Any(), 1) == 1
    >>> assert match(1, 1) == 1
    >>> assert match(1, 2) is NoMatch
    >>> assert match(1, 1, context={"x": 1}) == 1
    >>> assert match(1, 2, context={"x": 1}) is NoMatch
    >>> assert match([1, int], [1, 2]) == [1, 2]
    >>> assert match([1, int, "a" @ InstanceOf(str)], [1, 2, "three"]) == [
    ...     1,
    ...     2,
    ...     "three",
    ... ]
    """
    if context is None:
        context = {}

    pat = pattern(pat)
    result = pat.match(value, context)
    return NoMatch if result is NoMatch else result


IsTruish = Check(lambda x: bool(x))
IsNumber = InstanceOf(numbers.Number) & ~InstanceOf(bool)
IsString = InstanceOf(str)
