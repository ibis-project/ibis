from __future__ import annotations

import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Hashable
from weakref import WeakValueDictionary

from rich.console import Console

from ibis.common.caching import WeakCache
from ibis.common.validators import ImmutableProperty, Optional, Validator
from ibis.util import frozendict

EMPTY = inspect.Parameter.empty  # marker for missing argument

console = Console()


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


class Immutable(Hashable):

    __slots__ = ()

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )


class AnnotableMeta(BaseMeta):
    """
    Metaclass to turn class annotations into a validatable function signature.
    """

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct):
        # inherit from parent signatures
        params = {}
        properties = {}
        for parent in bases:
            try:
                params.update(parent.__signature__.parameters)
            except AttributeError:
                pass
            try:
                properties.update(parent.__properties__)
            except AttributeError:
                pass

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
            elif isinstance(attrib, ImmutableProperty):
                properties[name] = attrib
                slots.append(name)
            else:
                attribs[name] = attrib

        # mandatory fields without default values must preceed the optional
        # ones in the function signature, the partial ordering will be kept
        new_args, new_kwargs = [], []
        inherited_args, inherited_kwargs = [], []

        for name, param in params.items():
            if name in inherited:
                if param.default is EMPTY:
                    inherited_args.append(param)
                else:
                    inherited_kwargs.append(param)
            else:
                if param.default is EMPTY:
                    new_args.append(param)
                else:
                    new_kwargs.append(param)

        signature = inspect.Signature(
            inherited_args + new_args + new_kwargs + inherited_kwargs
        )

        attribs["__slots__"] = tuple(slots)
        attribs["__signature__"] = signature
        attribs["__properties__"] = properties
        attribs["argnames"] = tuple(signature.parameters.keys())
        return super().__new__(metacls, clsname, bases, attribs)


class Annotable(Base, Immutable, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    __slots__ = ("args", "_hash")

    @classmethod
    def __create__(cls, *args, **kwargs):
        bound = cls.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()

        # bound the signature to the passed arguments and apply the validators
        # before passing the arguments, so self.__init__() receives already
        # validated arguments as keywords
        kwargs = {}
        for name, value in bound.arguments.items():
            param = cls.__signature__.parameters[name]
            # TODO(kszucs): provide more error context on failure
            kwargs[name] = param.validate(kwargs, value)

        # construct the instance by passing the validated keyword arguments
        return super().__create__(**kwargs)

    def __init__(self, **kwargs):
        # set the already validated fields using object.__setattr__ since we
        # treat the annotable instances as immutable objects
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

        # optimizations to store frequently accessed generic properties
        args = tuple(kwargs[name] for name in self.argnames)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "_hash", hash((self.__class__, args)))

        # calculate special property-like objects only once due to the
        # immutable nature of annotable instances
        for name, prop in self.__properties__.items():
            object.__setattr__(self, name, prop(self))

        # any supplemental custom code provided by descendant classes
        self.__post_init__()

    def __post_init__(self):
        pass

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return super().__eq__(other)

    def __repr__(self) -> str:
        args = ", ".join(
            f"{name}={value!r}"
            for name, value in zip(self.argnames, self.args)
        )
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def _reconstruct(cls, kwargs):
        # bypass Annotable.__construct__() when deserializing
        self = cls.__new__(cls)
        self.__init__(**kwargs)
        return self

    def __reduce__(self):
        kwargs = dict(zip(self.argnames, self.args))
        return (self._reconstruct, (kwargs,))


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
