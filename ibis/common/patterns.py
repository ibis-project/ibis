from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from enum import Enum
from inspect import Parameter
from itertools import chain, zip_longest
from typing import Any as AnyType
from typing import (
    ForwardRef,
    Generic,  # noqa: F401
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import toolz
from typing_extensions import Annotated, GenericMeta, Self, get_args, get_origin

from ibis.common.collections import RewindableIterator, frozendict
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.typing import Sentinel, get_bound_typevars, get_type_params
from ibis.util import is_iterable, promote_tuple

try:
    from types import UnionType
except ImportError:
    UnionType = object()


T_cov = TypeVar("T_cov", covariant=True)


class CoercionError(Exception):
    ...


class ValidationError(Exception):
    ...


class MatchError(Exception):
    ...


class Coercible(ABC):
    """Protocol for defining coercible types.

    Coercible types define a special ``__coerce__`` method that accepts an object
    with an instance of the type. Used in conjunction with the ``coerced_to``
    pattern to coerce arguments to a specific type.
    """

    __slots__ = ()

    @classmethod
    @abstractmethod
    def __coerce__(cls, value: Any, **kwargs: Any) -> Self:
        ...


class Validator(ABC):
    __slots__ = ()

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
        pattern
            A pattern that matches the given type annotation.
        """
        # TODO(kszucs): cache the result of this function
        # TODO(kszucs): explore issubclass(typ, SupportsInt) etc.
        origin, args = get_origin(annot), get_args(annot)

        if origin is None:
            # the typehint is not generic
            if annot is Ellipsis or annot is AnyType:
                # treat both `Any` and `...` as wildcard
                return Any()
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
                    return Any()
            elif isinstance(annot, Enum):
                # for enums we check the value against the enum values
                return EqualTo(annot)
            elif isinstance(annot, (str, ForwardRef)):
                # for strings and forward references we check in a lazy way
                return LazyInstanceOf(annot)
            else:
                raise TypeError(f"Cannot create validator from annotation {annot!r}")
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
        elif issubclass(origin, Tuple):
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


class NoMatch(metaclass=Sentinel):
    """Marker to indicate that a pattern didn't match."""


# TODO(kszucs): have an As[int] or Coerced[int] type in ibis.common.typing which
# would be used to annotate an argument as coercible to int or to a certain type
# without needing for the type to inherit from Coercible
class Pattern(Validator, Hashable):
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
        match
            The result of the pattern matching. If the pattern doesn't match
            the value, then it must return the `NoMatch` sentinel value.
        """
        ...

    @abstractmethod
    def __eq__(self, other: Pattern) -> bool:
        ...

    def __invert__(self) -> Pattern:
        return Not(self)

    def __or__(self, other: Pattern) -> Pattern:
        return AnyOf(self, other)

    def __and__(self, other: Pattern) -> Pattern:
        return AllOf(self, other)

    def __rshift__(self, name: str) -> Pattern:
        return Capture(self, name)

    def __rmatmul__(self, name: str) -> Pattern:
        return Capture(self, name)

    def validate(
        self, value: AnyType, context: Optional[dict[str, AnyType]] = None
    ) -> Any:
        """Validate a value against the pattern.

        If the pattern doesn't match the value, then it raises a `ValidationError`.

        Parameters
        ----------
        value
            The value to match the pattern against.
        context
            A dictionary providing arbitrary context for the pattern matching.

        Returns
        -------
        match
            The matched / validated value.
        """
        result = self.match(value, context=context)
        if result is NoMatch:
            raise ValidationError(f"{value!r} doesn't match {self}")
        return result


class Matcher(Pattern):
    """A lightweight alternative to `ibis.common.grounds.Concrete`.

    This class is used to create immutable dataclasses with slots and a precomputed
    hash value for quicker dictionary lookups.
    """

    __slots__ = ("__precomputed_hash__",)

    def __init__(self, *args) -> Self:
        for name, value in zip_longest(self.__slots__, args):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "__precomputed_hash__", hash(args))

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name) for name in self.__slots__
        )

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __setattr__(self, name, value) -> None:
        raise AttributeError("Can't set attributes on immutable ENode instance")

    def __repr__(self):
        fields = {k: getattr(self, k) for k in self.__slots__}
        fieldstring = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fieldstring})"

    def __rich_repr__(self):
        for name in self.__slots__:
            yield name, getattr(self, name)


class Is(Matcher):
    """Pattern that matches a value against a reference value.

    Parameters
    ----------
    value
        The reference value to match against.
    """

    __slots__ = ("value",)

    def match(self, value, context):
        if value is self.value:
            return value
        else:
            return NoMatch


class Any(Matcher):
    """Pattern that accepts any value, basically a no-op."""

    __slots__ = ()

    def match(self, value, context):
        return value


class Capture(Matcher):
    """Pattern that captures a value in the context.

    Parameters
    ----------
    pattern
        The pattern to match against.
    name
        The name to use in the context if the pattern matches.
    """

    __slots__ = ("pattern", "name")

    def match(self, value, context):
        value = self.pattern.match(value, context=context)
        if value is NoMatch:
            return NoMatch
        context[self.name] = value
        return value


class Reference(Matcher):
    """Retrieve a value from the context.

    Parameters
    ----------
    key
        The key to retrieve from the state.
    """

    __slots__ = ("key",)

    def match(self, context):
        return context[self.key]


class Check(Matcher):
    """Pattern that checks a value against a predicate.

    Parameters
    ----------
    predicate
        The predicate to use.
    """

    __slots__ = ("predicate",)

    def match(self, value, context):
        if self.predicate(value):
            return value
        else:
            return NoMatch


class Apply(Matcher):
    """Pattern that applies a function to the value.

    Parameters
    ----------
    func
        The function to apply.
    """

    __slots__ = ("func",)

    def match(self, value, context):
        return self.func(value)


class Function(Matcher):
    """Pattern that checks a value against a function.

    Parameters
    ----------
    func
        The function to use.
    """

    __slots__ = ("func",)

    def match(self, value, context):
        return self.func(value, context)


class EqualTo(Matcher):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.
    """

    __slots__ = ("value",)

    def match(self, value, context):
        if value == self.value:
            return value
        else:
            return NoMatch


class Option(Matcher):
    """Pattern that matches `None` or a value that passes the inner validator.

    Parameters
    ----------
    pattern
        The inner pattern to use.
    """

    __slots__ = ("pattern", "default")

    def __init__(self, pattern, default=None):
        super().__init__(pattern, default)

    def match(self, value, context):
        if value is None:
            if self.default is None:
                return None
            else:
                return self.default
        else:
            return self.pattern.match(value, context=context)


class TypeOf(Matcher):
    """Pattern that matches a value that is of a given type."""

    __slots__ = ("type",)

    def match(self, value, context):
        if type(value) is self.type:
            return value
        else:
            return NoMatch


class SubclassOf(Matcher):
    """Pattern that matches a value that is a subclass of a given type.

    Parameters
    ----------
    type
        The type to check against.
    """

    __slots__ = ("type",)

    def match(self, value, context):
        if issubclass(value, self.type):
            return value
        else:
            return NoMatch


class InstanceOf(Matcher):
    """Pattern that matches a value that is an instance of a given type.

    Parameters
    ----------
    types
        The type to check against.
    """

    __slots__ = ("type",)

    def match(self, value, context):
        if isinstance(value, self.type):
            return value
        else:
            return NoMatch


class GenericInstanceOf(Matcher):
    """Pattern that matches a value that is an instance of a given generic type.

    Parameters
    ----------
    typ
        The type to check against (must be a generic type).

    Examples
    --------
    >>> class MyNumber(Generic[T_cov]):
    ...    value: T_cov
    ...
    ...    def __init__(self, value: T_cov):
    ...        self.value = value
    ...
    ...    def __eq__(self, other):
    ...        return type(self) is type(other) and self.value == other.value
    ...
    >>> p = GenericInstanceOf(MyNumber[int])
    >>> assert p.match(MyNumber(1), {}) == MyNumber(1)
    >>> assert p.match(MyNumber(1.0), {}) is NoMatch
    >>>
    >>> p = GenericInstanceOf(MyNumber[float])
    >>> assert p.match(MyNumber(1.0), {}) == MyNumber(1.0)
    >>> assert p.match(MyNumber(1), {}) is NoMatch
    """

    __slots__ = ("origin", "field_patterns")

    def __init__(self, typ):
        origin = get_origin(typ)
        typevars = get_bound_typevars(typ)

        field_patterns = {}
        for var, (attr, type_) in typevars.items():
            if not var.__covariant__:
                raise TypeError(
                    f"Typevar {var} is not covariant, cannot use it in a GenericInstanceOf"
                )
            field_patterns[attr] = Pattern.from_typehint(type_, allow_coercion=False)

        super().__init__(origin, frozendict(field_patterns))

    def match(self, value, context):
        if not isinstance(value, self.origin):
            return NoMatch

        for field, pattern in self.field_patterns.items():
            attr = getattr(value, field)
            if pattern.match(attr, context) is NoMatch:
                return NoMatch

        return value


class LazyInstanceOf(Matcher):
    """A version of `InstanceOf` that accepts qualnames instead of imported classes.

    Useful for delaying imports.

    Parameters
    ----------
    types
        The types to check against.
    """

    __slots__ = ("types", "check")

    def __init__(self, types):
        types = promote_tuple(types)
        check = lazy_singledispatch(lambda x: False)
        check.register(types, lambda x: True)
        super().__init__(promote_tuple(types), check)

    def match(self, value, *, context):
        if self.check(value):
            return value
        else:
            return NoMatch


# TODO(kszucs): to support As[int] or CoercedTo[int] syntax
class CoercedTo(Matcher):
    """Force a value to have a particular Python type.

    If a Coercible subclass is passed, the `__coerce__` method will be used to
    coerce the value. Otherwise, the type will be called with the value as the
    only argument.

    Parameters
    ----------
    type
        The type to coerce to.
    """

    __slots__ = ("target",)

    def __new__(cls, target):
        if issubclass(target, Coercible):
            return super().__new__(cls)
        else:
            return Apply(target)

    def match(self, value, context):
        try:
            value = self.target.__coerce__(value)
        except CoercionError:
            return NoMatch

        if isinstance(value, self.target):
            return value
        else:
            return NoMatch

    def __repr__(self):
        return f"CoercedTo({self.target.__name__!r})"


As = CoercedTo


class GenericCoercedTo(Matcher):
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

    def __init__(self, target):
        # TODO(kszucs): when constructing the checker we shouldn't allow
        # coercions, only type checks
        origin = get_origin(target)
        checker = GenericInstanceOf(target)
        params = frozendict(get_type_params(target))
        super().__init__(origin, params, checker)

    def match(self, value, context):
        try:
            value = self.origin.__coerce__(value, **self.params)
        except CoercionError:
            return NoMatch

        if self.checker.match(value, context) is NoMatch:
            return NoMatch

        return value


class Not(Matcher):
    """Pattern that matches a value that does not match a given pattern.

    Parameters
    ----------
    pattern
        The pattern which the value should not match.
    """

    __slots__ = ("pattern",)

    def match(self, value, context):
        if self.pattern.match(value, context=context) is NoMatch:
            return value
        else:
            return NoMatch


class AnyOf(Matcher):
    """Pattern that if any of the given patterns match.

    Parameters
    ----------
    patterns
        The patterns to match against. The first pattern that matches will be
        returned.
    """

    __slots__ = ("patterns",)

    def __init__(self, *patterns):
        super().__init__(patterns)

    def match(self, value, context):
        for pattern in self.patterns:
            result = pattern.match(value, context=context)
            if result is not NoMatch:
                return result
        return NoMatch


class AllOf(Matcher):
    """Pattern that matches if all of the given patterns match.

    Parameters
    ----------
    patterns
        The patterns to match against. The value will be passed through each
        pattern in order. The changes applied to the value propagate through the
        patterns.
    """

    __slots__ = ("patterns",)

    def __init__(self, *patterns):
        super().__init__(patterns)

    def match(self, value, context):
        for pattern in self.patterns:
            value = pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
        return value


class Length(Matcher):
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
        super().__init__(at_least, at_most)

    def match(self, value, *, context):
        length = len(value)
        if self.at_least is not None and length < self.at_least:
            return NoMatch
        if self.at_most is not None and length > self.at_most:
            return NoMatch
        return value


class Contains(Matcher):
    """Pattern that matches if a value contains a given value.

    Parameters
    ----------
    needle
        The item that the passed value should contain.
    """

    __slots__ = ("needle",)

    def match(self, value, context):
        if self.needle in value:
            return value
        else:
            return NoMatch


class IsIn(Matcher):
    """Pattern that matches if a value is in a given set.

    Parameters
    ----------
    haystack
        The set of values that the passed value should be in.
    """

    __slots__ = ("haystack",)

    def __init__(self, haystack):
        super().__init__(frozenset(haystack))

    def match(self, value, context):
        if value in self.haystack:
            return value
        else:
            return NoMatch


In = IsIn


class SequenceOf(Matcher):
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

    __slots__ = ("item_pattern", "type_pattern", "length_pattern")

    def __init__(
        self,
        item: Pattern,
        type: type = tuple,
        exactly: Optional[int] = None,
        at_least: Optional[int] = None,
        at_most: Optional[int] = None,
    ):
        item_pattern = pattern(item)
        type_pattern = CoercedTo(type)
        length_pattern = Length(at_least=at_least, at_most=at_most)
        super().__init__(item_pattern, type_pattern, length_pattern)

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        result = []
        for value in values:
            value = self.item_pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        result = self.type_pattern.match(result, context=context)
        if result is NoMatch:
            return NoMatch

        return self.length_pattern.match(result, context=context)


class TupleOf(Matcher):
    """Pattern that matches if the respective items in a tuple match the given patterns.

    Parameters
    ----------
    fields
        The patterns to match the respective items in the tuple.
    """

    __slots__ = ("field_patterns",)

    def __new__(cls, fields):
        if isinstance(fields, tuple):
            return super().__new__(cls)
        else:
            return SequenceOf(fields, tuple)

    def match(self, values, context):
        if not is_iterable(values):
            return NoMatch

        if len(values) != len(self.field_patterns):
            return NoMatch

        result = []
        for pattern, value in zip(self.field_patterns, values):
            value = pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        return tuple(result)


class MappingOf(Matcher):
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

    __slots__ = ("key_pattern", "value_pattern", "type_pattern")

    def __init__(self, key: Pattern, value: Pattern, type: type = dict):
        super().__init__(pattern(key), pattern(value), CoercedTo(type))

    def match(self, value, context):
        if not isinstance(value, Mapping):
            return NoMatch

        result = {}
        for k, v in value.items():
            if (k := self.key_pattern.match(k, context=context)) is NoMatch:
                return NoMatch
            if (v := self.value_pattern.match(v, context=context)) is NoMatch:
                return NoMatch
            result[k] = v

        result = self.type_pattern.match(result, context=context)
        if result is NoMatch:
            return NoMatch

        return result


class Attrs(Matcher):
    __slots__ = ("patterns",)

    def __init__(self, **patterns):
        super().__init__(frozendict(toolz.valmap(pattern, patterns)))

    def match(self, value, context):
        for attr, pattern in self.patterns.items():
            if not hasattr(value, attr):
                return NoMatch

            v = getattr(value, attr)
            if match(pattern, v, context=context) is NoMatch:
                return NoMatch

        return value


class Object(Matcher):
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

    __slots__ = ("type", "attrs_pattern")

    def __init__(self, type, *args, **kwargs):
        kwargs.update(dict(zip(type.__match_args__, args)))
        super().__init__(type, Attrs(**kwargs))

    def match(self, value, context):
        if not isinstance(value, self.type):
            return NoMatch

        if not self.attrs_pattern.match(value, context=context):
            return NoMatch

        return value


class CallableWith(Matcher):
    __slots__ = ("arg_patterns", "return_pattern")

    def __init__(self, args, return_=None):
        super().__init__(tuple(args), return_ or Any())

    def match(self, value, context):
        from ibis.common.annotations import annotated

        if not callable(value):
            return NoMatch

        fn = annotated(self.arg_patterns, self.return_pattern, value)

        has_varargs = False
        positional, keyword_only = [], []
        for p in fn.__signature__.parameters.values():
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                positional.append(p)
            elif p.kind is Parameter.KEYWORD_ONLY:
                keyword_only.append(p)
            elif p.kind is Parameter.VAR_POSITIONAL:
                has_varargs = True

        if keyword_only:
            raise MatchError(
                "Callable has mandatory keyword-only arguments which cannot be specified"
            )
        elif len(positional) > len(self.arg_patterns):
            # Callable has more positional arguments than expected")
            return NoMatch
        elif len(positional) < len(self.arg_patterns) and not has_varargs:
            # Callable has less positional arguments than expected")
            return NoMatch
        else:
            return fn


class PatternSequence(Matcher):
    __slots__ = ("pattern_window",)

    def __init__(self, patterns):
        current_patterns = [
            SequenceOf(Any()) if p is Ellipsis else pattern(p) for p in patterns
        ]
        following_patterns = chain(current_patterns[1:], [Not(Any())])
        pattern_window = tuple(zip(current_patterns, following_patterns))
        super().__init__(pattern_window)

    @property
    def first_pattern(self):
        return self.pattern_window[0][0]

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

            if isinstance(current, (SequenceOf, PatternSequence)):
                if isinstance(following, SequenceOf):
                    following = following.item_pattern
                elif isinstance(following, PatternSequence):
                    following = following.first_pattern

                matches = []
                while True:
                    it.checkpoint()
                    try:
                        item = next(it)
                    except StopIteration:
                        break

                    if match(following, item, context) is NoMatch:
                        matches.append(item)
                    else:
                        it.rewind()
                        break

                res = original.match(matches, context=context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.extend(res)
            else:
                try:
                    item = next(it)
                except StopIteration:
                    return NoMatch

                res = original.match(item, context=context)
                if res is NoMatch:
                    return NoMatch
                else:
                    result.append(res)

        return result


class PatternMapping(Matcher):
    __slots__ = ("keys_pattern", "values_pattern")

    def __init__(self, patterns):
        keys_pattern = PatternSequence(list(map(pattern, patterns.keys())))
        values_pattern = PatternSequence(list(map(pattern, patterns.values())))
        super().__init__(keys_pattern, values_pattern)

    def match(self, value, context):
        if not isinstance(value, Mapping):
            return NoMatch

        keys = value.keys()
        if (keys := self.keys_pattern.match(keys, context=context)) is NoMatch:
            return NoMatch

        values = value.values()
        if (values := self.values_pattern.match(values, context=context)) is NoMatch:
            return NoMatch

        return dict(zip(keys, values))


class Between(Matcher):
    """Match a value between two bounds.

    Parameters
    ----------
    lower
        The lower bound.
    upper
        The upper bound.
    """

    __slots__ = ("lower", "upper")

    def __init__(self, lower: float = -math.inf, upper: float = math.inf):
        super().__init__(lower, upper)

    def match(self, value, context):
        if self.lower <= value <= self.upper:
            return value
        else:
            return NoMatch


IsTruish = Check(lambda x: bool(x))
IsNumber = InstanceOf(numbers.Number) & ~InstanceOf(bool)
IsString = InstanceOf(str)


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
    >>>
    >>> assert as_int.match(1, {}) == 1

    Returns
    -------
    pattern
        The constructed pattern.
    """
    if obj is Ellipsis:
        return Any()
    elif isinstance(obj, Pattern):
        return obj
    elif isinstance(obj, Mapping):
        return PatternMapping(obj)
    elif isinstance(obj, type):
        return InstanceOf(obj)
    elif get_origin(obj):
        return Pattern.from_typehint(obj)
    elif is_iterable(obj):
        return PatternSequence(obj)
    elif callable(obj):
        return Function(obj)
    else:
        return EqualTo(obj)


def match(pat: Pattern, value: AnyType, context: Optional[dict[str, AnyType]] = None):
    """Match a value against a pattern.

    Parameters
    ----------
    pat
        The pattern to match against.
    value
        The value to match.
    context
        Arbitrary mapping of values to be used while matching.

    Examples
    --------
    >>> assert match(Any(), 1) == {}
    >>> assert match(1, 1) == {}
    >>> assert match(1, 2) is NoMatch
    >>> assert match(1, 1, context={"x": 1}) == {"x": 1}
    >>> assert match(1, 2, context={"x": 1}) is NoMatch
    >>> assert match([1, int], [1, 2]) == {}
    >>> assert match([1, int, "a" @ InstanceOf(str)], [1, 2, "three"]) == {"a": "three"}
    """
    if context is None:
        context = {}

    pat = pattern(pat)
    if pat.match(value, context=context) is NoMatch:
        return NoMatch

    return context


class Topmost(Matcher):
    """Traverse the value tree topmost first and match the first value that matches."""

    __slots__ = ("searcher", "filter")

    def __init__(self, searcher, filter=None):
        super().__init__(pattern(searcher), filter)

    def match(self, value, context):
        result = self.searcher.match(value, context)
        if result is not NoMatch:
            return result

        for child in value.__children__(self.filter):
            result = self.match(child, context)
            if result is not NoMatch:
                return result

        return NoMatch


class Innermost(Matcher):
    """Traverse the value tree innermost first and match the first value that matches."""

    __slots__ = ("searcher", "filter")

    def __init__(self, searcher, filter=None):
        super().__init__(pattern(searcher), filter)

    def match(self, value, context):
        for child in value.__children__(self.filter):
            result = self.match(child, context)
            if result is not NoMatch:
                return result

        return self.searcher.match(value, context)
