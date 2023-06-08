from __future__ import annotations

import math
from contextlib import suppress
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Container,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import toolz
from typing_extensions import Annotated, get_args, get_origin

from ibis.common.collections import FrozenDict
from ibis.common.dispatch import lazy_singledispatch
from ibis.common.exceptions import IbisTypeError
from ibis.common.patterns import Coercible
from ibis.util import flatten_iterable, is_function, is_iterable

try:
    from types import UnionType
except ImportError:
    UnionType = object()

K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


class Validator(Callable):
    """Abstract base class for defining argument validators."""

    __slots__ = ()

    @classmethod
    def from_typehint(cls, annot: type) -> Validator:
        """Construct a validator from a python type annotation.

        Parameters
        ----------
        annot
            The typehint annotation to construct a validator from.

        Returns
        -------
        validator
            A validator that can be used to validate objects, typically function
            arguments.
        """
        # TODO(kszucs): cache the result of this function
        origin, args = get_origin(annot), get_args(annot)

        if origin is None:
            if annot is Any:
                return any_
            elif isinstance(annot, TypeVar):
                return any_
            elif issubclass(annot, Coercible):
                return coerced_to(annot)
            else:
                return instance_of(annot)
        elif origin is Literal:
            return isin(args)
        elif origin is UnionType or origin is Union:
            inners = map(cls.from_typehint, args)
            return any_of(tuple(inners))
        elif origin is Annotated:
            annot, *extras = args
            return all_of((instance_of(annot), *extras))
        elif issubclass(origin, Tuple):
            first, *rest = args
            if rest == [Ellipsis]:
                inners = cls.from_typehint(first)
            else:
                inners = tuple(map(cls.from_typehint, args))
            return tuple_of(inners, type=coerced_to(origin))
        elif issubclass(origin, Sequence):
            (value_inner,) = map(cls.from_typehint, args)
            return sequence_of(value_inner, type=coerced_to(origin))
        elif issubclass(origin, Mapping):
            key_inner, value_inner = map(cls.from_typehint, args)
            return mapping_of(key_inner, value_inner, type=coerced_to(origin))
        elif issubclass(origin, Callable):
            if args:
                arg_inners = tuple(map(cls.from_typehint, args[0]))
                return_inner = cls.from_typehint(args[1])
                return callable_with(arg_inners, return_inner)
            else:
                return instance_of(Callable)
        else:
            raise NotImplementedError(
                f"Cannot create validator from annotation {annot} {origin}"
            )


# TODO(kszucs): in order to cache valiadator instances we could subclass
# grounds.Singleton, but the imports would need to be reorganized
class Curried(toolz.curry, Validator):
    """Enable convenient validator definition by decorating plain functions."""

    def __repr__(self):
        return '{}({}{})'.format(
            self.func.__name__,
            repr(self.args)[1:-1],
            ', '.join(f'{k}={v!r}' for k, v in self.keywords.items()),
        )


validator = Curried


@validator
def ref(key: str, *, this: Mapping[str, Any]) -> Any:
    """Retrieve a value from the already validated state.

    Parameters
    ----------
    key
        The key to retrieve from the state.
    this
        The state to retrieve the value from, usually the result of an annotated
        function signature validation (including annotable object creation).

    Returns
    -------
    value
        The value retrieved from the state.
    """
    try:
        return this[key]
    except KeyError:
        raise IbisTypeError(f"Could not get `{key}` from {this}")


@validator
def any_(arg: Any, **kwargs: Any) -> Any:
    """Validator that accepts any value, basically a no-op."""
    return arg


@validator
def option(inner: Validator, arg: Any, *, default: Any = None, **kwargs) -> Any:
    """Validator that accepts `None` or a value that passes the inner validator.

    Parameters
    ----------
    inner
        The inner validator to use.
    arg
        The value to validate.
    default
        The default value to use if `arg` is `None`.
    kwargs
        Additional keyword arguments to pass to the inner validator.

    Returns
    -------
    validated
        The validated value or the default value if `arg` is `None`.
    """
    if arg is None:
        if default is None:
            return None
        elif is_function(default):
            arg = default()
        else:
            arg = default
    return inner(arg, **kwargs)


@validator
def instance_of(klasses: type | tuple[type], arg: Any, **kwargs: Any) -> Any:
    """Require that a value has a particular Python type.

    Parameters
    ----------
    klasses
        The type or tuple of types to validate against.
    arg
        The value to validate.
    kwargs
        Omitted keyword arguments.

    Returns
    -------
    validated
        The input argument if it is an instance of the given type(s).
    """
    if not isinstance(arg, klasses):
        # TODO(kszucs): unify errors coming from various validators
        raise IbisTypeError(
            f"Given argument with type {type(arg)} is not an instance of {klasses}"
        )
    return arg


@validator
def equal_to(value: T, arg: T, **kwargs: Any) -> T:
    """Require that a value is equal to a particular value."""
    if arg != value:
        raise IbisTypeError(f"Given argument {arg} is not equal to {value}")
    return arg


@validator
def coerced_to(klass: T, arg: Any, **kwargs: Any) -> T:
    """Force a value to have a particular Python type.

    If a Coercible subclass is passed, the `__coerce__` method will be used to
    coerce the value. Otherwise, the type will be called with the value as the
    only argument.

    Parameters
    ----------
    klass
        The type to coerce to.
    arg
        The value to coerce.
    kwargs
        Additional keyword arguments to pass to the inner validator.

    Returns
    -------
    validated
        The coerced value which is checked to be an instance of the given type.
    """
    if isinstance(arg, klass):
        return arg
    try:
        arg = klass.__coerce__(arg)
    except AttributeError:
        arg = klass(arg)
    return instance_of(klass, arg, **kwargs)


class lazy_instance_of(Validator):
    """A version of `instance_of` that accepts qualnames instead of imported classes.

    Useful for delaying imports.
    """

    def __init__(self, classes):
        classes = (classes,) if isinstance(classes, str) else tuple(classes)
        self._classes = classes
        self._check = lazy_singledispatch(lambda x: False)
        self._check.register(classes, lambda x: True)

    def __repr__(self):
        return f"lazy_instance_of(classes={self._classes!r})"

    def __call__(self, arg, **kwargs):
        if self._check(arg):
            return arg
        raise IbisTypeError(
            f"Given argument with type {type(arg)} is not an instance of "
            f"{self._classes}"
        )


@validator
def any_of(inners: Iterable[Validator], arg: Any, **kwargs: Any) -> Any:
    """At least one of the inner validators must pass.

    Parameters
    ----------
    inners
        Iterable of value validators, each of which is applied from left to right and
        the first one that passes gets returned.
    arg
        Value to be validated.
    kwargs
        Keyword arguments

    Returns
    -------
    arg : Any
        Value maybe coerced by inner validators to the appropriate types
    """
    for inner in inners:
        with suppress(IbisTypeError, ValueError):
            return inner(arg, **kwargs)

    raise IbisTypeError(
        "argument passes none of the following rules: "
        f"{', '.join(map(repr, inners))}"
    )


one_of = any_of


@validator
def all_of(inners: Iterable[Validator], arg: Any, **kwargs: Any) -> Any:
    """Construct a validator of other valdiators.

    Parameters
    ----------
    inners
        Iterable of value validators, each of which is applied from left to
        right so `allof([rule1, rule2], arg)` is the same as `rule2(rule1(arg))`.
    arg
        Value to be validated.
    kwargs
        Keyword arguments

    Returns
    -------
    arg : Any
      Value maybe coerced by inner validators to the appropriate types
    """
    for inner in inners:
        arg = inner(arg, **kwargs)
    return arg


@validator
def isin(values: Container, arg: T, **kwargs: Any) -> T:
    """Check if the value is in the given container.

    Parameters
    ----------
    values
        Container of values to check against.
    arg
        Value to be looked for.
    kwargs
        Omitted keyword arguments.

    Returns
    -------
    validated
        The input argument if it is in the given container.
    """
    if arg not in values:
        raise ValueError(f'Value with type {type(arg)} is not in {values!r}')
    return arg


@validator
def map_to(mapping: Mapping[K, V], variant: K, **kwargs: Any) -> V:
    """Check if the value is in the given mapping and return the corresponding value.

    Parameters
    ----------
    mapping
        Mapping of values to check against.
    variant
        Value to be looked for.
    kwargs
        Omitted keyword arguments.

    Returns
    -------
    validated
        The value corresponding to the input argument if it is in the given mapping.
    """
    try:
        return mapping[variant]
    except KeyError:
        raise ValueError(f'Value with type {type(variant)} is not in {mapping!r}')


@validator
def pair_of(
    inner1: Validator, inner2: Validator, arg: Any, *, type=tuple, **kwargs
) -> tuple[Any, Any]:
    """Validate a pair of values (tuple of 2 items).

    Parameters
    ----------
    inner1
        Validator to apply to the first element of the pair.
    inner2
        Validator to apply to the second element of the pair.
    arg
        Pair to validate.
    type
        Type to coerce the pair to, typically a tuple.
    kwargs
        Additional keyword arguments to pass to the inner validator.

    Returns
    -------
    validated
        The validated pair with each element coerced according to the inner validators.
    """
    try:
        first, second = arg
    except KeyError:
        raise IbisTypeError('Argument must be a pair')
    return type((inner1(first, **kwargs), inner2(second, **kwargs)))


@validator
def sequence_of(
    inner: Validator,
    arg: Any,
    *,
    type: Callable[[Iterable], T],
    length: int | None = None,
    min_length: int = 0,
    max_length: int = math.inf,
    flatten: bool = False,
    **kwargs: Any,
) -> T:
    """Validate a sequence of values.

    Parameters
    ----------
    inner
        Validator to apply to each element of the sequence.
    arg
        Sequence to validate.
    type
        Type to coerce the sequence to, typically a tuple or list.
    length
        If specified, the sequence must have exactly this length.
    min_length
        The sequence must have at least this many elements.
    max_length
        The sequence must have at most this many elements.
    flatten
        If True, the sequence is flattened before validation.
    kwargs
        Keyword arguments to pass to the inner validator.

    Returns
    -------
    validated
        The coerced sequence containing validated elements.
    """
    if not is_iterable(arg):
        raise IbisTypeError('Argument must be a sequence')

    if length is not None:
        min_length = max_length = length
    if len(arg) < min_length:
        raise IbisTypeError(f'Arg must have at least {min_length} number of elements')
    if len(arg) > max_length:
        raise IbisTypeError(f'Arg must have at most {max_length} number of elements')

    if flatten:
        arg = flatten_iterable(arg)

    return type(inner(item, **kwargs) for item in arg)


@validator
def tuple_of(inner: Validator | tuple[Validator], arg: Any, *, type=tuple, **kwargs):
    """Validate a tuple of values.

    Parameters
    ----------
    inner
        Either a balidator to apply to each element of the tuple or a tuple of
        validators which are applied to the elements of the tuple in order.
    arg
        Sequence to validate.
    type
        Type to coerce the sequence to, a tuple by default.
    kwargs
        Keyword arguments to pass to the inner validator.

    Returns
    -------
    validated
        The coerced tuple containing validated elements.
    """
    if isinstance(inner, tuple):
        if is_iterable(arg):
            arg = tuple(arg)
        else:
            raise IbisTypeError('Argument must be a sequence')

        if len(inner) != len(arg):
            raise IbisTypeError(f'Argument must has length {len(inner)}')

        return type(validator(item, **kwargs) for validator, item in zip(inner, arg))
    else:
        return sequence_of(inner, arg, type=type, **kwargs)


@validator
def mapping_of(
    key_inner: Validator,
    value_inner: Validator,
    arg: Any,
    *,
    type: T,
    **kwargs: Any,
) -> T:
    """Validate a mapping of values.

    Parameters
    ----------
    key_inner
        Validator to apply to each key of the mapping.
    value_inner
        Validator to apply to each value of the mapping.
    arg
        Mapping to validate.
    type
        Type to coerce the mapping to, typically a dict.
    kwargs
        Keyword arguments to pass to the inner validator.

    Returns
    -------
    validated
        The coerced mapping containing validated keys and values.
    """
    if not isinstance(arg, Mapping):
        raise IbisTypeError('Argument must be a mapping')
    return type(
        {key_inner(k, **kwargs): value_inner(v, **kwargs) for k, v in arg.items()}
    )


@validator
def callable_with(
    arg_inners: Sequence[Validator],
    return_inner: Validator,
    value: Any,
    **kwargs: Any,
) -> Callable:
    """Validate a callable with a given signature and return type.

    The rule's responsility is twofold:
    1. Validate the signature of the callable (keyword only arguments are not supported)
    2. Wrap the callable with validation logic that validates the arguments and the
       return value at runtime.

    Parameters
    ----------
    arg_inners
        Sequence of validators to apply to the arguments of the callable.
    return_inner
        Validator to apply to the return value of the callable.
    value
        Callable to validate.
    kwargs
        Keyword arguments to pass to the inner validators.

    Returns
    -------
    validated
        The callable wrapped with validation logic.
    """
    from ibis.common.annotations import annotated

    if not callable(value):
        raise IbisTypeError("Argument must be a callable")

    fn = annotated(arg_inners, return_inner, value)

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
        raise IbisTypeError(
            "Callable has mandatory keyword-only arguments which cannot be specified"
        )
    elif len(positional) > len(arg_inners):
        raise IbisTypeError("Callable has more positional arguments than expected")
    elif len(positional) < len(arg_inners) and not has_varargs:
        raise IbisTypeError("Callable has less positional arguments than expected")
    else:
        return fn


@validator
def int_(arg: Any, min: int = 0, max: int = math.inf, **kwargs: Any) -> int:
    """Validate an integer.

    Parameters
    ----------
    arg
        Integer to validate.
    min
        Minimum value of the integer.
    max
        Maximum value of the integer.
    kwargs
        Omitted keyword arguments.

    Returns
    -------
    validated
        The validated integer.
    """
    if not isinstance(arg, int):
        raise IbisTypeError('Argument must be an integer')
    arg = min_(min, arg, **kwargs)
    arg = max_(max, arg, **kwargs)
    return arg


@validator
def min_(min: int, arg: int, **kwargs: Any) -> int:
    if arg < min:
        raise ValueError(f'Argument must be greater than {min}')
    return arg


@validator
def max_(max: int, arg: int, **kwargs: Any) -> int:
    if arg > max:
        raise ValueError(f'Argument must be less than {max}')
    return arg


str_ = instance_of(str)
bool_ = instance_of(bool)
none_ = instance_of(type(None))
dict_of = mapping_of(type=dict)
list_of = sequence_of(type=list)
frozendict_of = mapping_of(type=FrozenDict)
