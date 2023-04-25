from __future__ import annotations

import contextlib
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Any
from weakref import WeakValueDictionary

from ibis.common.annotations import EMPTY, Argument, Attribute, Signature, attribute
from ibis.common.caching import WeakCache
from ibis.common.collections import FrozenDict
from ibis.common.typing import evaluate_annotations
from ibis.common.validators import Validator


class BaseMeta(ABCMeta):
    __slots__ = ()

    def __new__(metacls, clsname, bases, dct, **kwargs):
        # enforce slot definitions
        dct.setdefault("__slots__", ())
        return super().__new__(metacls, clsname, bases, dct, **kwargs)

    def __call__(cls, *args, **kwargs) -> Base:
        return cls.__create__(*args, **kwargs)


class Base(metaclass=BaseMeta):
    __slots__ = ('__weakref__',)
    __create__ = classmethod(type.__call__)


class AnnotableMeta(BaseMeta):
    """Metaclass to turn class annotations into a validatable function signature."""

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct, **kwargs):
        # inherit signature from parent classes
        signatures, attributes = [], {}
        for parent in bases:
            with contextlib.suppress(AttributeError):
                attributes.update(parent.__attributes__)
            with contextlib.suppress(AttributeError):
                signatures.append(parent.__signature__)

        # collection type annotations and convert them to validators
        module_name = dct.get('__module__')
        annotations = dct.get('__annotations__', {})
        typehints = evaluate_annotations(annotations, module_name)
        for name, typehint in typehints.items():
            validator = Validator.from_typehint(typehint)
            if name in dct:
                dct[name] = Argument.default(dct[name], validator, typehint=typehint)
            else:
                dct[name] = Argument.required(validator, typehint=typehint)

        # collect the newly defined annotations
        slots = list(dct.pop('__slots__', []))
        namespace, arguments = {}, {}
        for name, attrib in dct.items():
            if isinstance(attrib, Validator):
                attrib = Argument.required(attrib)

            if isinstance(attrib, Argument):
                arguments[name] = attrib
                attributes[name] = attrib
                slots.append(name)
            elif isinstance(attrib, Attribute):
                attributes[name] = attrib
                slots.append(name)
            else:
                namespace[name] = attrib

        # merge the annotations with the parent annotations
        signature = Signature.merge(*signatures, **arguments)
        argnames = tuple(signature.parameters.keys())

        namespace.update(
            __argnames__=argnames,
            __attributes__=attributes,
            __match_args__=argnames,
            __signature__=signature,
            __slots__=tuple(slots),
        )
        return super().__new__(metacls, clsname, bases, namespace, **kwargs)


class Annotable(Base, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    @classmethod
    def __create__(cls, *args, **kwargs) -> Annotable:
        # construct the instance by passing the validated keyword arguments
        kwargs = cls.__signature__.validate(*args, **kwargs)
        return super().__create__(**kwargs)

    @classmethod
    def __recreate__(cls, kwargs) -> Annotable:
        # bypass signature binding by requiring keyword arguments only
        kwargs = cls.__signature__.validate_nobind(**kwargs)
        return super().__create__(**kwargs)

    def __init__(self, **kwargs) -> None:
        # set the already validated arguments
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

        # post-initialize the remaining attributes
        for name, field in self.__attributes__.items():
            if isinstance(field, Attribute):
                if (value := field.initialize(self)) is not EMPTY:
                    object.__setattr__(self, name, value)

    def __setattr__(self, name, value) -> None:
        if field := self.__attributes__.get(name):
            value = field.validate(value, this=self)
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        args = (f"{n}={getattr(self, n)!r}" for n in self.__argnames__)
        argstring = ", ".join(args)
        return f"{self.__class__.__name__}({argstring})"

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        return all(
            getattr(self, name, None) == getattr(other, name, None)
            for name in self.__attributes__
        )

    @property
    def __args__(self):
        return tuple(getattr(self, name) for name in self.__argnames__)

    def copy(self, **overrides: Any) -> Annotable:
        """Return a copy of this object with the given overrides.

        Parameters
        ----------
        overrides
            Argument override values

        Returns
        -------
        Annotable
            New instance of the copied object
        """
        this = copy(self)
        for name, value in overrides.items():
            setattr(this, name, value)
        return this


class Immutable(Base):
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __setattr__(self, name: str, _: Any) -> None:
        raise AttributeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )


class Singleton(Base):
    __instances__ = WeakValueDictionary()

    @classmethod
    def __create__(cls, *args, **kwargs) -> Singleton:
        key = (cls, args, FrozenDict(kwargs))
        try:
            return cls.__instances__[key]
        except KeyError:
            instance = super().__create__(*args, **kwargs)
            cls.__instances__[key] = instance
            return instance


class Comparable(Base):
    __cache__ = WeakCache()

    def __eq__(self, other) -> bool:
        try:
            return self.__cached_equals__(other)
        except TypeError:
            return NotImplemented

    @abstractmethod
    def __equals__(self, other) -> bool:
        ...

    def __cached_equals__(self, other) -> bool:
        if self is other:
            return True

        # type comparison should be cheap
        if type(self) is not type(other):
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
    """Opinionated base class for immutable data classes."""

    @attribute.default
    def __args__(self):
        return tuple(getattr(self, name) for name in self.__argnames__)

    @attribute.default
    def __precomputed_hash__(self):
        return hash((self.__class__, self.__args__))

    def __reduce__(self):
        # assuming immutability and idempotency of the __init__ method, we can
        # reconstruct the instance from the arguments without additional attributes
        state = dict(zip(self.__argnames__, self.__args__))
        return (self.__recreate__, (state,))

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

    def copy(self, **overrides):
        kwargs = dict(zip(self.__argnames__, self.__args__))
        kwargs.update(overrides)
        return self.__recreate__(kwargs)
