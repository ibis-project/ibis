from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from typing import Any
from weakref import WeakValueDictionary

from rich.console import Console

from ibis.common.caching import WeakCache
from ibis.common.validators import (
    ImmutableProperty,
    Parameter,
    Signature,
    Validator,
)
from ibis.util import frozendict

EMPTY = inspect.Parameter.empty  # marker for missing argument

console = Console()


class BaseMeta(ABCMeta):

    __slots__ = ()

    def __call__(cls, *args, **kwargs):
        return cls.__create__(*args, **kwargs)


class Base(metaclass=BaseMeta):

    __slots__ = ('__weakref__',)

    @classmethod
    def __create__(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)


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

        signature = Signature(
            inherited_args + new_args + new_kwargs + inherited_kwargs
        )
        argnames = tuple(signature.parameters.keys())

        attribs["__slots__"] = tuple(slots)
        attribs["__signature__"] = signature
        attribs["__properties__"] = properties
        attribs["__argnames__"] = argnames
        attribs["__match_args__"] = argnames
        return super().__new__(metacls, clsname, bases, attribs)


class Annotable(Base, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    @classmethod
    def __create__(cls, *args, **kwargs):
        # construct the instance by passing the validated keyword arguments
        kwargs = cls.__signature__.validate(*args, **kwargs)
        return super().__create__(**kwargs)

    def __init__(self, **kwargs):
        # set the already validated fields using object.__setattr__
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        # allow child classes to do some post-initialization
        self.__post_init__()

    def __post_init__(self):
        # calculate special property-like objects only once due to the
        # immutable nature of annotable instances
        for name, prop in self.__properties__.items():
            object.__setattr__(self, name, prop(self))

    def __setattr__(self, name, value):
        param = self.__signature__.parameters[name]
        if param.default is not None or value is not None:
            value = param.validate(value, this=self.__getstate__())
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        args = (f"{n}={getattr(self, n)!r}" for n in self.__argnames__)
        argstring = ", ".join(args)
        return f"{self.__class__.__name__}({argstring})"

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        return all(
            getattr(self, n) == getattr(other, n) for n in self.__argnames__
        )

    def __getstate__(self):
        return {name: getattr(self, name) for name in self.__argnames__}

    def __setstate__(self, state):
        self.__init__(**state)

    def copy(self, **overrides):
        kwargs = self.__getstate__()
        kwargs.update(overrides)
        return self.__class__(**kwargs)


class Immutable(Base):
    __slots__ = ()

    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )


class Singleton(Base):

    __slots__ = ()
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


class Comparable(Base):

    __slots__ = ()
    __cache__ = WeakCache()

    def __eq__(self, other):
        try:
            return self.__cached_equals__(other)
        except TypeError:
            return NotImplemented  # noqa: F901

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
        if id(self) < id(other):
            key = (self, other)
        else:
            key = (other, self)

        try:
            result = self.__cache__[key]
        except KeyError:
            result = self.__equals__(other)
            self.__cache__[key] = result

        return result


class Concrete(Immutable, Comparable, Annotable):

    __slots__ = ("__args__", "__precomputed_hash__")

    def __post_init__(self):
        # optimizations to store frequently accessed generic properties
        arguments = tuple(getattr(self, name) for name in self.__argnames__)
        hashvalue = hash((self.__class__, arguments))
        object.__setattr__(self, "__args__", arguments)
        object.__setattr__(self, "__precomputed_hash__", hashvalue)
        super().__post_init__()

    def __hash__(self):
        return self.__precomputed_hash__

    def __equals__(self, other):
        return self.__args__ == other.__args__

    @property
    def args(self):
        return self.__args__

    @property
    def argnames(self):
        return self.__argnames__
