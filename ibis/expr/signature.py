from __future__ import annotations

import inspect
from abc import ABCMeta
from typing import Any, Callable, Hashable, MutableMapping

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


@util.deprecated(version="3.0", instead="use Validator if needed")
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

    def __new__(metacls, clsname, bases, dct):
        params = {}
        for parent in bases:
            # inherit from parent signatures
            if hasattr(parent, '__signature__'):
                params.update(parent.__signature__.parameters)

        slots = list(dct.pop('__slots__', []))
        attribs = {}
        for name, attrib in dct.items():
            if isinstance(attrib, Validator):
                # so we can set directly
                params[name] = Parameter(name, validator=attrib)
                slots.append(name)
            else:
                attribs[name] = attrib

        # mandatory fields without default values must preceed the optional
        # ones in the function signature, the partial ordering will be kept
        params = sorted(
            params.values(), key=lambda p: p.default is EMPTY, reverse=True
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


class Annotable(Hashable, util.EqMixin, metaclass=AnnotableMeta):
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

    def _type_check(self, other: Any) -> None:
        if type(self) != type(other):
            raise TypeError(
                "invalid equality comparison between "
                f"{type(self)} and {type(other)}"
            )

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

    def __component_eq__(
        self,
        other: Annotable,
        cache: MutableMapping[Hashable, bool],
    ) -> bool:
        return util.seq_eq(self.args, other.args, cache=cache)

    def __getstate__(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.argnames}

    def flat_args(self):
        import ibis.expr.schema as sch

        for arg in self.args:
            if not isinstance(arg, sch.Schema) and util.is_iterable(arg):
                yield from arg
            else:
                yield arg
