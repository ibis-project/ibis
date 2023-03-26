"""Converted the previous validator system into a pattern matching system.

The previously used validator system had the following problems:
- Used curried functions which was error prone to missing arguments and was hard to debug.
- Used exceptions for control flow which raising exceptions from the innermost function call giving cryptic error messages.
- While it was similarly composable it was hard to see whether the validator was fully constructed or not.
- We wasn't able traverse the nested validators due to the curried functions.

In addition to those the new approach has the following advantages:
- The pattern matching system is fully composable.
- Not we support syntax sugar for combining patterns using & and |.
- We can capture values at mutiple levels using either `>>` or `@` syntax.
- Support structural pattern matching for sequences, mappings and objects.
"""
from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from inspect import Parameter
from itertools import chain, zip_longest
from typing import (
    Any as AnyType,
)
from typing import (
    Callable,
    Literal,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Annotated, get_args, get_origin

from ibis.common.collections import RewindableIterator, frozendict
from ibis.common.dispatch import lazy_singledispatch
from ibis.util import is_function, is_iterable, promote_tuple

try:
    from types import UnionType
except ImportError:
    UnionType = object()


class MatchError(Exception):
    ...


class ValidationError(Exception):
    ...


class NoMatch:
    """Sentinel value for when a pattern doesn't match."""


class Coercible(ABC):
    """Protocol for defining coercible types.

    Coercible types define a special ``__coerce__`` method that accepts an object
    with an instance of the type. Used in conjunction with the ``coerced_to``
    pattern to coerce arguments to a specific type.
    """

    __slots__ = ()

    @classmethod
    @abstractmethod
    def __coerce__(cls, obj):
        ...


class Pattern(ABC, Hashable):
    @abstractmethod
    def match(self, value, *, context):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    # TODO(kszucs): consider to add match_strict() with a default implementation
    # that calls match()

    def __invert__(self):
        return Not(self)

    def __or__(self, other):
        return AnyOf(self, other)

    def __and__(self, other):
        return AllOf(self, other)

    def __rshift__(self, name):
        return Capture(self, name)

    def __rmatmul__(self, name):
        return Capture(self, name)

    def validate(self, value, *, context):
        if self.match(value, context=context) is NoMatch:
            raise ValidationError(f"{value} doesn't match {self}")

    @classmethod
    def from_typehint(cls, annot: type) -> Pattern:
        """Construct a pattern from a python type annotation.

        Parameters
        ----------
        annot
            The typehint annotation to construct the pattern from. If a string is
            available then it must be evaluated before passing it to this function
            using the ``evaluate_typehint`` function.
        module
            The module to evaluate the type annotation in. This is only used
            when the first argument is a string (or ForwardRef).

        Returns
        -------
        pattern
            A pattern that matches the given type annotation.
        """
        # TODO(kszucs): cache the result of this function
        origin, args = get_origin(annot), get_args(annot)

        if origin is None:
            if annot is AnyType:
                return Any()
            elif isinstance(annot, TypeVar):
                return Any()
            elif issubclass(annot, Coercible):
                return CoercedTo(annot)
            else:
                return InstanceOf(annot)
        elif origin is Literal:
            return IsIn(args)
        elif origin is UnionType or origin is Union:
            inners = map(cls.from_typehint, args)
            return AnyOf(*inners)
        elif origin is Annotated:
            annot, *extras = args
            return AllOf(InstanceOf(annot), *extras)
        elif issubclass(origin, Tuple):
            first, *rest = args
            if rest == [Ellipsis]:
                inners = cls.from_typehint(first)
            else:
                inners = tuple(map(cls.from_typehint, args))
            return TupleOf(inners)
        elif issubclass(origin, Sequence):
            (value_inner,) = map(cls.from_typehint, args)
            return SequenceOf(value_inner, type=origin)
            # return SequenceOf(value_inner, type=CoercedTo(origin))
        elif issubclass(origin, Mapping):
            key_inner, value_inner = map(cls.from_typehint, args)
            return MappingOf(key_inner, value_inner, type=origin)
        elif issubclass(origin, Callable):
            if args:
                arg_inners = tuple(map(cls.from_typehint, args[0]))
                return_inner = cls.from_typehint(args[1])
                return CallableWith(arg_inners, return_inner)
            else:
                return InstanceOf(Callable)
        else:
            raise NotImplementedError(
                f"Cannot create validator from annotation {annot} {origin}"
            )


# extend Singleton base class to make this a singleton
class Matcher(Pattern):
    """A lightweight alternative to `ibis.common.grounds.Concrete`.

    This class is used to create immutable dataclasses with slots and a precomputed
    hash value for quicker dictionary lookups.
    """

    __slots__ = ("__precomputed_hash__",)

    def __init__(self, *args):
        for name, value in zip_longest(self.__slots__, args):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "__precomputed_hash__", hash(args))

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        for name in self.__slots__:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")

    def __repr__(self):
        fields = {k: getattr(self, k) for k in self.__slots__}
        fieldstring = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fieldstring})"

    def __rich_repr__(self):
        for name in self.__slots__:
            yield name, getattr(self, name)


class Any(Matcher):
    """Pattern that accepts any value, basically a no-op."""

    __slots__ = ()

    def match(self, value, *, context):
        return value


class Capture(Matcher):
    __slots__ = ("pattern", "name")

    def match(self, value, *, context):
        value = self.pattern.match(value, context=context)
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

    def match(self, *, context):
        return context[self.key]


class Check(Matcher):
    """Pattern that checks a value against a predicate.

    Parameters
    ----------
    predicate
        The predicate to use.
    """

    __slots__ = ("predicate",)

    def match(self, value, *, context):
        if self.predicate(value):
            return value
        else:
            return NoMatch


class EqualTo(Matcher):
    """Pattern that checks a value equals to the given value.

    Parameters
    ----------
    value
        The value to check against.
    """

    __slots__ = ("value",)

    def match(self, value, *, context):
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
    default
        The default value to use if `arg` is `None`.
    """

    __slots__ = ("pattern", "default")

    def __init__(self, pattern, default=None):
        super().__init__(pattern, default)

    def match(self, value, *, context):
        default = self.default
        if value is None:
            if self.default is None:
                return None
            elif is_function(default):
                value = default()
            else:
                value = default
        return self.pattern.match(value, context=context)


class TypeOf(Matcher):
    """Pattern that matches a value that is of a given type."""

    __slots__ = ("type",)

    def match(self, value, *, context):
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

    def match(self, value, *, context):
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

    __slots__ = ("types",)

    def match(self, value, *, context):
        if isinstance(value, self.types):
            return value
        else:
            return NoMatch


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

    __slots__ = ("type",)

    def match(self, value, *, context):
        klass = self.type
        if isinstance(value, type):
            return value

        try:
            return klass.__coerce__(value)
        except AttributeError:
            return klass(value)


class Not(Matcher):
    __slots__ = ("pattern",)

    def match(self, value, *, context):
        if self.pattern.match(value, context=context) is NoMatch:
            return value
        else:
            return NoMatch


class AnyOf(Matcher):
    __slots__ = ("patterns",)

    def __init__(self, *patterns):
        super().__init__(patterns)

    def match(self, value, *, context):
        for pattern in self.patterns:
            if pattern.match(value, context=context) is not NoMatch:
                return value
        return NoMatch


class AllOf(Matcher):
    __slots__ = ("patterns",)

    def __init__(self, *patterns):
        super().__init__(patterns)

    def match(self, value, *, context):
        for pattern in self.patterns:
            value = pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
        return value


def NoneOf(*args) -> Pattern:
    """Match none of the passed patterns."""
    return Not(AnyOf(*args))


class Length(Matcher):
    __slots__ = ("at_least", "at_most")

    def __init__(self, exactly=None, at_least=None, at_most=None):
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
    __slots__ = ("needle",)

    def match(self, value, *, context):
        if self.needle in value:
            return value
        else:
            return NoMatch


class IsIn(Matcher):
    __slots__ = ("haystack",)

    def __init__(self, haystack):
        super().__init__(frozenset(haystack))

    def match(self, value, *, context):
        if value in self.haystack:
            return value
        else:
            return NoMatch


class SequenceOf(Matcher):
    __slots__ = ("pattern", "type", "length_pattern")

    def __init__(self, pat, type=tuple, at_least=None, at_most=None):
        super().__init__(pattern(pat), type, Length(at_least=at_least, at_most=at_most))

    def match(self, values, *, context):
        if not is_iterable(values):
            return NoMatch

        result = []
        for value in values:
            value = self.pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        result = self.type(result)

        return self.length_pattern.match(result, context=context)


class TupleOf(Matcher):
    __slots__ = ("patterns",)

    def __new__(cls, patterns):
        if isinstance(patterns, tuple):
            return super().__new__(cls)
        else:
            return SequenceOf(patterns, tuple)

    def match(self, values, *, context):
        if not is_iterable(values):
            return NoMatch

        if len(values) != len(self.patterns):
            return NoMatch

        result = []
        for pattern, value in zip(self.patterns, values):
            value = pattern.match(value, context=context)
            if value is NoMatch:
                return NoMatch
            result.append(value)

        return tuple(result)


class MappingOf(Matcher):
    __slots__ = ("key_pattern", "value_pattern", "type")

    def __init__(self, key_pattern, value_pattern, type=dict):
        super().__init__(pattern(key_pattern), pattern(value_pattern), type)

    def match(self, value, *, context):
        if not isinstance(value, Mapping):
            return NoMatch

        result = {}
        for k, v in value.items():
            if (k := self.key_pattern.match(k, context=context)) is NoMatch:
                return NoMatch
            if (v := self.value_pattern.match(v, context=context)) is NoMatch:
                return NoMatch
            result[k] = v

        return self.type(result)


class Object(Matcher):
    __slots__ = ("type", "patterns")

    def __init__(self, type, *args, **kwargs):
        match_args = getattr(type, "__match_args__", tuple())
        kwargs.update(dict(zip(args, match_args)))
        super().__init__(type, frozendict(kwargs))

    def match(self, value, *, context):
        if not isinstance(value, self.type):
            return NoMatch

        for attr, pattern in self.patterns.items():
            if not hasattr(value, attr):
                return NoMatch

            if match(pattern, getattr(value, attr), context=context) is NoMatch:
                return NoMatch

        return value


class CallableWith(Matcher):
    __slots__ = ("arg_patterns", "return_pattern")

    def __init__(self, arg_patterns, return_pattern=None):
        super().__init__(tuple(arg_patterns), return_pattern or Any())

    def match(self, value, *, context):
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
    __slots__ = ("pattern_pairs",)

    def __init__(self, patterns):
        current_patterns = [
            SequenceOf(Any()) if p is Ellipsis else pattern(p) for p in patterns
        ]
        following_patterns = chain(current_patterns[1:], [Not(Any())])
        pattern_pairs = tuple(zip(current_patterns, following_patterns))
        super().__init__(pattern_pairs)

    def match(self, value, *, context):
        it = RewindableIterator(value)
        result = []

        if not self.pattern_pairs:
            try:
                next(it)
            except StopIteration:
                return result
            else:
                return NoMatch

        for current, following in self.pattern_pairs:
            original = current

            if isinstance(current, Capture):
                current = current.pattern
            if isinstance(following, Capture):
                following = following.pattern

            if isinstance(current, (SequenceOf, PatternSequence)):
                if isinstance(following, SequenceOf):
                    following = following.pattern
                elif isinstance(following, PatternSequence):
                    following = following.pattern_pairs[0][0]

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

    def match(self, value, *, context):
        if not isinstance(value, Mapping):
            return NoMatch

        keys = value.keys()
        if (keys := self.keys_pattern.match(keys, context=context)) is NoMatch:
            return NoMatch

        values = value.values()
        if (values := self.values_pattern.match(values, context=context)) is NoMatch:
            return NoMatch

        return dict(zip(keys, values))


IsTruish = Check(lambda x: bool(x))
IsNumber = InstanceOf(numbers.Number) & ~InstanceOf(bool)
IsString = InstanceOf(str)


def ListOf(pattern):
    return SequenceOf(pattern, type=list)


def DictOf(key_pattern, value_pattern):
    return MappingOf(key_pattern, value_pattern, type=dict)


def FrozenDictOf(key_pattern, value_pattern):
    return MappingOf(key_pattern, value_pattern, type=frozendict)


def pattern(obj):
    """Create a pattern from an object."""
    if isinstance(obj, Pattern):
        return obj
    elif isinstance(obj, Mapping):
        return PatternMapping(obj)
    elif is_iterable(obj):
        return PatternSequence(obj)
    elif obj is Ellipsis:
        return Any()
    else:
        return EqualTo(obj)


def match(pat, value, context=None):
    if context is None:
        context = {}

    pat = pattern(pat)
    if pat.match(value, context=context) is NoMatch:
        return NoMatch

    return context


# TODO(kszucs): add a ChildOf matcher to match against the children of a node
