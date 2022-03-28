from __future__ import annotations

import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Hashable
from weakref import WeakValueDictionary

from ibis.util import frozendict

from .. import util
from .caching import WeakCache
from .validators import Optional, Validator

EMPTY = inspect.Parameter.empty  # marker for missing argument


class BaseMeta(ABCMeta):

    __slots__ = ()

    def __call__(cls, *args, **kwargs):
        return cls.__create__(*args, **kwargs)


class Base(metaclass=BaseMeta):

    __slots__ = ()

    @classmethod
    def __create__(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)


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


class AnnotableMeta(BaseMeta):
    """
    Metaclass to turn class annotations into a validatable function signature.
    """

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct):
        # inherit from parent signatures
        params = {}
        for parent in bases:
            try:
                signature = parent.__signature__
            except AttributeError:
                pass
            else:
                params.update(signature.parameters)
        # store the originally inherited keys so we can reorder later
        inherited = set(params.keys())

        # collect the newly defined parameters
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
        new_params, inherited_params, optional_inherited_params = [], [], []
        for name, param in params.items():
            if name in inherited:
                if param.default is EMPTY:
                    inherited_params.append(param)
                else:
                    optional_inherited_params.append(param)
            else:
                new_params.append(param)

        signature = inspect.Signature(
            inherited_params + new_params + optional_inherited_params
        )

        attribs["__slots__"] = tuple(slots)
        attribs["__signature__"] = signature
        attribs["argnames"] = tuple(signature.parameters.keys())
        return super().__new__(metacls, clsname, bases, attribs)


class Annotable(Base, Hashable, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    __slots__ = ("args", "_hash")

    @classmethod
    def __create__(cls, *args, **kwargs):
        bound = cls.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()

        kwargs = {}
        for name, value in bound.arguments.items():
            param = cls.__signature__.parameters[name]
            # TODO(kszucs): provide more error context on failure
            kwargs[name] = param.validate(kwargs, value)

        instance = super().__create__(**kwargs)
        instance.__post_init__()
        return instance

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

    def __eq__(self, other):
        return super().__eq__(other)

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )

    def __repr__(self) -> str:
        args = ", ".join(
            f"{name}={value!r}"
            for name, value in zip(self.argnames, self.args)
            if not name.startswith("_")
        )
        return f"{self.__class__.__name__}({args})"

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

    def flat_args(self):
        import ibis.expr.schema as sch

        for arg in self.args:
            if not isinstance(arg, sch.Schema) and util.is_iterable(arg):
                yield from arg
            else:
                yield arg


class Singleton(Base):

    __slots__ = ('__weakref__',)
    __instances__ = WeakValueDictionary()

    @classmethod
    def __create__(cls, *args, **kwargs):
        key = (cls, args, frozendict(kwargs))
        try:
            return cls.__instances__[key]
        except KeyError:
            instance = super().__create__(*args, **kwargs)
            cls.__instances__[key] = instance
            return instance


class Comparable(ABC):

    __slots__ = ('__weakref__',)
    __cache__ = WeakCache()

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        try:
            return self.__cached_equals__(other)
        except TypeError:
            raise NotImplemented  # noqa: F901

    @abstractmethod
    def __equals__(self, other):
        ...

    def __cached_equals__(self, other):
        if self is other:
            return True

        # type comparison should be cheap
        if type(self) != type(other):
            return False

        # reduce space required for commutative operation
        if hash(self) < hash(other):
            key = (self, other)
        else:
            key = (other, self)

        try:
            result = self.__cache__[key]
        except KeyError:
            result = self.__equals__(other)
            self.__cache__[key] = result

        return result
