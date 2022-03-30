from __future__ import annotations

import inspect
from contextlib import suppress

from ..common.exceptions import IbisTypeError
from ..util import flatten_iterable, is_function, is_iterable

EMPTY = inspect.Parameter.empty  # marker for missing argument


class Validator:
    __slots__ = ()

    def validate(self, arg, **kwargs):
        raise NotImplementedError()


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

    __slots__ = (
        'validator',
        'default',
    )

    def __init__(self, validator, default=None):
        self.validator = validator
        self.default = default

    def __repr__(self):
        return f"{self.__class__.__name__}({self.validator!r}, {self.default!r})"

    def validate(self, arg, **kwargs):
        if arg is None:
            if self.default is None:
                return None
            elif is_function(self.default):
                arg = self.default()
            else:
                arg = self.default

        return self.validator.validate(arg, **kwargs)


class CurriedValidator(Validator):

    __slots__ = ("func", "args", "kwargs")

    def __init__(self, func, args=None, kwargs=None):
        # TODO(kszucs): validate that func is callable
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}

    def __call__(self, *args, **kwargs):
        new_args = self.args + args
        new_kwargs = {**self.kwargs, **kwargs}
        return self.__class__(self.func, new_args, new_kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.func!r}, {self.args!r}, {self.kwargs!r})"
        )

    def validate(self, arg, **kwargs):
        return self.func(*self.args, arg, **self.kwargs, **kwargs)


optional = Optional
validator = CurriedValidator


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
            return inner.validate(arg, **kwargs)

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
        arg = inner.validate(arg, this=this)
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

    return type(inner.validate(item, **kwargs) for item in arg)


list_of = container_of(type=list)
tuple_of = container_of(type=tuple)

# TODO(kszucs): try to cache validator objects
# TODO(kszucs): try a quicker curry implementation
