from __future__ import annotations

import inspect
from contextlib import suppress
from typing import Callable

import toolz

from ibis.common.exceptions import IbisTypeError
from ibis.util import flatten_iterable, is_function, is_iterable

EMPTY = inspect.Parameter.empty  # marker for missing argument


class Validator(Callable):
    """
    Abstract base class for defining argument validators.
    """


class Optional(Validator):
    """
    Validator to allow missing arguments.

    Parameters
    ----------
    validator : Validator
        Used to do the actual validation if the argument gets passed.
    default : Any, default None
        Value to return with in case of a missing argument.
    """

    __slots__ = ('validator', 'default')

    def __init__(self, validator, default=None):
        self.validator = validator
        self.default = default

    def __call__(self, arg, **kwargs):
        if arg is None:
            if self.default is None:
                return None
            elif is_function(self.default):
                arg = self.default()
            else:
                arg = self.default

        return self.validator(arg, **kwargs)


class Curried(toolz.curry, Validator):
    """
    Enable convenient validator definition by decorating plain functions.
    """

    def __repr__(self):
        return '{}({}{})'.format(
            self.func.__name__,
            repr(self.args)[1:-1],
            ', '.join(f'{k}={v!r}' for k, v in self.keywords.items()),
        )


class ImmutableProperty(Callable):
    """
    Abstract base class for defining stored properties.
    """

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, instance):
        return self.fn(instance)


# aliases for convenience
optional = Optional
validator = Curried
immutable_property = ImmutableProperty


@validator
def noop(arg, **kwargs):
    return arg


@validator
def instance_of(klasses, arg, **kwargs):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klasses):
        raise IbisTypeError(
            f'Given argument with type {type(arg)} '
            f'is not an instance of {klasses}'
        )
    return arg


@validator
def one_of(inners, arg, **kwargs):
    """At least one of the inner validators must pass"""
    for inner in inners:
        with suppress(IbisTypeError, ValueError):
            return inner(arg, **kwargs)

    raise IbisTypeError(
        "argument passes none of the following rules: "
        f"{', '.join(map(repr, inners))}"
    )


@validator
def compose_of(inners, arg, *, this):
    """All of the inner validators must pass.

    The order of inner validators matters.

    Parameters
    ----------
    inners : List[validator]
      Functions are applied from right to left so allof([rule1, rule2], arg) is
      the same as rule1(rule2(arg)).
    arg : Any
      Value to be validated.

    Returns
    -------
    arg : Any
      Value maybe coerced by inner validators to the appropiate types
    """
    for inner in reversed(inners):
        arg = inner(arg, this=this)
    return arg


@validator
def isin(values, arg, **kwargs):
    if arg not in values:
        raise ValueError(f'Value with type {type(arg)} is not in {values!r}')
    if isinstance(values, dict):  # TODO check for mapping instead
        return values[arg]
    else:
        return arg


@validator
def map_to(mapping, variant, **kwargs):
    try:
        return mapping[variant]
    except KeyError:
        raise ValueError(
            f'Value with type {type(variant)} is not in {mapping!r}'
        )


@validator
def container_of(inner, arg, *, type, min_length=0, flatten=False, **kwargs):
    if not is_iterable(arg):
        raise IbisTypeError('Argument must be a sequence')

    if len(arg) < min_length:
        raise IbisTypeError(
            f'Arg must have at least {min_length} number of elements'
        )

    if flatten:
        arg = flatten_iterable(arg)

    return type(inner(item, **kwargs) for item in arg)


list_of = container_of(type=list)
tuple_of = container_of(type=tuple)

# TODO(kszucs): try to cache validator objects
# TODO(kszucs): try a quicker curry implementation
