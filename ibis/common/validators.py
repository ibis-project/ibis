from __future__ import annotations

import math
from contextlib import suppress
from typing import Any, Callable, Iterable, Union

import toolz
from typing_extensions import Annotated, get_args, get_origin

from ibis.common.dispatch import lazy_singledispatch
from ibis.common.exceptions import IbisTypeError
from ibis.common.typing import evaluate_typehint
from ibis.util import flatten_iterable, is_function, is_iterable


class Validator(Callable):
    """Abstract base class for defining argument validators."""

    __slots__ = ()

    @classmethod
    def from_annotation(cls, annot, module=None):
        # TODO(kszucs): cache the result of this function
        annot = evaluate_typehint(annot, module)
        origin_type = get_origin(annot)

        if origin_type is Union:
            inners = map(cls.from_annotation, get_args(annot))
            return any_of(tuple(inners))
        elif origin_type is list:
            (inner,) = map(cls.from_annotation, get_args(annot))
            return list_of(inner)
        elif origin_type is tuple:
            (inner,) = map(cls.from_annotation, get_args(annot))
            return tuple_of(inner)
        elif origin_type is dict:
            key_type, value_type = map(cls.from_annotation, get_args(annot))
            return dict_of(key_type, value_type)
        elif origin_type is Annotated:
            annot, *extras = get_args(annot)
            return all_of((instance_of(annot), *extras))
        elif annot is Any:
            return any_
        else:
            return instance_of(annot)


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
def ref(key, *, this):
    try:
        return this[key]
    except KeyError:
        raise IbisTypeError(f"Could not get `{key}` from {this}")


@validator
def any_(arg, **kwargs):
    return arg


@validator
def option(inner, arg, *, default=None, **kwargs):
    if arg is None:
        if default is None:
            return None
        elif is_function(default):
            arg = default()
        else:
            arg = default
    return inner(arg, **kwargs)


@validator
def instance_of(klasses, arg, **kwargs):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klasses):
        # TODO(kszucs): unify errors coming from various validators
        raise IbisTypeError(
            f"Given argument with type {type(arg)} is not an instance of {klasses}"
        )
    return arg


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
def any_of(inners, arg, **kwargs):
    """At least one of the inner validators must pass."""
    for inner in inners:
        with suppress(IbisTypeError, ValueError):
            return inner(arg, **kwargs)

    raise IbisTypeError(
        "argument passes none of the following rules: "
        f"{', '.join(map(repr, inners))}"
    )


one_of = any_of


@validator
def all_of(inners: Iterable[validator], arg: Any, **kwargs: Any) -> Any:
    """Construct a validator of other valdiators.

    Parameters
    ----------
    inners
        Iterable of function Functions, each of which is applied from right to
        left so `allof([rule1, rule2], arg)` is the same as `rule1(rule2(arg))`.
    arg
        Value to be validated.
    kwargs
        Keyword arguments

    Returns
    -------
    arg : Any
      Value maybe coerced by inner validators to the appropiate types
    """
    for inner in inners:
        arg = inner(arg, **kwargs)
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
        raise ValueError(f'Value with type {type(variant)} is not in {mapping!r}')


@validator
def container_of(inner, arg, *, type, min_length=0, flatten=False, **kwargs):
    if not is_iterable(arg):
        raise IbisTypeError('Argument must be a sequence')

    if len(arg) < min_length:
        raise IbisTypeError(f'Arg must have at least {min_length} number of elements')

    if flatten:
        arg = flatten_iterable(arg)

    return type(inner(item, **kwargs) for item in arg)


@validator
def mapping_of(key_inner, value_inner, arg, *, type, **kwargs):
    return type(
        (key_inner(k, **kwargs), value_inner(v, **kwargs)) for k, v in arg.items()
    )


@validator
def int_(arg, min=0, max=math.inf, **kwargs):
    if not isinstance(arg, int):
        raise IbisTypeError('Argument must be an integer')
    if arg < min:
        raise ValueError(f'Argument must be greater than {min}')
    if arg > max:
        raise ValueError(f'Argument must be less than {max}')
    return arg


@validator
def min_(min, arg, **kwargs):
    if arg < min:
        raise ValueError(f'Argument must be greater than {min}')
    return arg


str_ = instance_of(str)
bool_ = instance_of(bool)
none_ = instance_of(type(None))
dict_of = mapping_of(type=dict)
list_of = container_of(type=list)
tuple_of = container_of(type=tuple)
