from __future__ import annotations

import inspect
from abc import ABCMeta
from contextlib import suppress
from typing import Any, Callable, Hashable

from toolz import curry

import ibis.common.exceptions as com
from ibis import util

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
            elif util.is_function(self.default):
                arg = self.default()
            else:
                arg = self.default

        return self.validator(arg, **kwargs)


class validator(curry, Validator):
    """
    Enable convenient validator definition by decorating plain functions.
    """

    def __repr__(self):
        return '{}({}{})'.format(
            self.func.__name__,
            repr(self.args)[1:-1],
            ', '.join(f'{k}={v!r}' for k, v in self.keywords.items()),
        )


optional = Optional


@validator
def instance_of(klasses, arg, **kwargs):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klasses):
        raise com.IbisTypeError(
            f'Given argument with type {type(arg)} '
            f'is not an instance of {klasses}'
        )
    return arg


@validator
def noop(arg, **kwargs):
    return arg


@validator
def one_of(inners, arg, **kwargs):
    """At least one of the inner validators must pass"""
    for inner in inners:
        with suppress(com.IbisTypeError, ValueError):
            return inner(arg, **kwargs)

    raise com.IbisTypeError(
        "argument passes none of the following rules: "
        f"{', '.join(map(repr, inners))}"
    )


@validator
def compose_of(inners, arg, **kwargs):
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
        raise ValueError(f'Unexpected value `{variant}`')


@validator
def container_of(inner, arg, *, type, min_length=0, flatten=False, **kwargs):
    if not util.is_iterable(arg):
        raise com.IbisTypeError('Argument must be a sequence')

    if len(arg) < min_length:
        raise com.IbisTypeError(
            f'Arg must have at least {min_length} number of elements'
        )

    if flatten:
        arg = util.flatten_iterable(arg)

    return type(inner(item, **kwargs) for item in arg)


# TODO(kszucs): remove list_of rule eventually
list_of = tuple_of = container_of(type=tuple)


class Parameter(inspect.Parameter):
    """
    Augmented Parameter class to additionally hold a validator object.
    """

    __slots__ = ('_validator',)

    def __init__(
        self,
        name,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        *,
        validator=EMPTY,
    ):
        super().__init__(
            name,
            kind,
            default=None if isinstance(validator, Optional) else EMPTY,
        )
        self._validator = validator

    @property
    def validator(self):
        return self._validator

    def validate(self, this, arg):
        if self.validator is EMPTY:
            return arg
        else:
            # TODO(kszucs): use self._validator
            return self.validator(arg, this=this)


class _ValidatorFunction(Validator):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _InstanceOf(Validator):
    def __init__(self, typ):
        self.typ = typ

    def __call__(self, arg, **kwargs):
        if not isinstance(arg, self.typ):
            raise TypeError(self.typ)
        return arg


@util.deprecated(version="3.0", instead="Validator")
def Argument(validator, default=EMPTY):
    """Argument constructor
    Parameters
    ----------
    validator : Union[Callable[[arg], coerced], Type, Tuple[Type]]
        Function which handles validation and/or coercion of the given
        argument.
    default : Union[Any, Callable[[], str]]
        In case of missing (None) value for validation this will be used.
        Note, that default value (except for None) must also pass the inner
        validator.
        If callable is passed, it will be executed just before the inner,
        and itsreturn value will be treaded as default.
    """
    if isinstance(validator, Validator):
        pass
    elif isinstance(validator, type):
        validator = _InstanceOf(validator)
    elif isinstance(validator, tuple):
        assert util.all_of(validator, type)
        validator = _InstanceOf(validator)
    elif isinstance(validator, Validator):
        validator = validator
    elif callable(validator):
        validator = _ValidatorFunction(validator)
    else:
        raise TypeError(
            'Argument validator must be a callable, type or '
            'tuple of types, given: {}'.format(validator)
        )

    if default is EMPTY:
        return validator
    else:
        return Optional(validator, default=default)


class AnnotableMeta(ABCMeta):
    """
    Metaclass to turn class annotations into a validatable function signature.
    """

    # TODO(kszucs): add tests for argument order
    def __new__(metacls, clsname, bases, dct):
        inherited = {}
        for parent in bases:
            # inherit from parent signatures
            if hasattr(parent, '__signature__'):
                inherited.update(parent.__signature__.parameters)

        slots = list(dct.pop('__slots__', []))
        params = {}
        attribs = {}
        for name, attrib in dct.items():
            if isinstance(attrib, Validator):
                # so we can set directly
                params[name] = Parameter(name, validator=attrib)
                inherited.pop(name, None)
                slots.append(name)
            else:
                attribs[name] = attrib

        # mandatory fields without default values must preceed the optional
        # ones in the function signature, the partial ordering will be kept
        params = sorted(
            list(params.values()) + list(inherited.values()),
            key=lambda p: p.default is EMPTY,
            reverse=True,
        )

        attribs["__slots__"] = tuple(slots)
        attribs["__signature__"] = inspect.Signature(params)
        attribs["argnames"] = tuple(attribs["__signature__"].parameters.keys())

        return super().__new__(metacls, clsname, bases, attribs)

    def __call__(cls, *args, **kwargs):
        bound = cls.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()

        kwargs = {}
        for name, value in bound.arguments.items():
            param = cls.__signature__.parameters[name]
            kwargs[name] = param.validate(kwargs, value)

        instance = super().__call__(**kwargs)
        instance.__post_init__()
        return instance


# util.CachedEqMixin,
class Annotable(Hashable, util.CachedEqMixin, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    __slots__ = "args", "_hash"

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

    # TODO(kszucs): split for better __init__ composability but we can
    # directly call it from __init__ too
    def __post_init__(self):
        args = tuple(getattr(self, name) for name in self.argnames)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "_hash", hash((type(self), args)))

    def __hash__(self):
        return self._hash

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )

    @classmethod
    def _reconstruct(cls, kwargs):
        # bypass AnnotableMeta.__call__() when deserializing
        self = cls.__new__(cls)
        self.__init__(**kwargs)
        self.__post_init__()
        return self

    def __reduce__(self):
        kwargs = dict(zip(self.argnames, self.args))
        return (self._reconstruct, (kwargs,))

    def __equals__(self, other):
        return type(self) == type(other) and self.args == other.args

    def __getstate__(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.argnames}

    def flat_args(self):
        for arg in self.args:
            if util.is_iterable(arg):
                yield from arg
            else:
                yield arg

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state after unpickling.

        Parameters
        ----------
        state
            A dictionary storing the objects attributes.
        """
        self._args = None
        self._hash = None
        for key, value in state.items():
            setattr(self, key, value)
