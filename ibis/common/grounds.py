from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any
from weakref import WeakValueDictionary

from ibis.common.annotations import (
    Argument,
    Attribute,
    Default,
    Initialized,
    Mandatory,
    Signature,
)
from ibis.common.caching import WeakCache
from ibis.common.typing import evaluate_typehint
from ibis.common.validators import Validator
from ibis.util import frozendict


class BaseMeta(ABCMeta):

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct, **kwargs):
        # enforce slot definitions
        dct.setdefault("__slots__", ())
        return super().__new__(metacls, clsname, bases, dct, **kwargs)

    def __call__(cls, *args, **kwargs):
        return cls.__create__(*args, **kwargs)


class Base(metaclass=BaseMeta):

    __slots__ = ('__weakref__',)

    @classmethod
    def __create__(cls, *args, **kwargs):
        return type.__call__(cls, *args, **kwargs)


class AnnotableMeta(BaseMeta):
    """Metaclass to turn class annotations into a validatable function
    signature."""

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct, **kwargs):
        # inherit signature from parent classes
        signatures, attributes = [], {}
        for parent in bases:
            try:
                attributes.update(parent.__attributes__)
            except AttributeError:
                pass
            try:
                signatures.append(parent.__signature__)
            except AttributeError:
                pass

        # collection type annotations and convert them to validators
        module = dct.get('__module__')
        annots = dct.get('__annotations__', {})
        for name, annot in annots.items():
            typehint = evaluate_typehint(annot, module)
            validator = Validator.from_annotation(typehint)
            if name in dct:
                dct[name] = Default(validator=validator, default=dct[name])
            else:
                dct[name] = Mandatory(validator=validator)

        # collect the newly defined annotations
        slots = list(dct.pop('__slots__', []))
        attribs, args = {}, {}
        for name, attrib in dct.items():
            if isinstance(attrib, Validator):
                attrib = Mandatory(attrib)

            if isinstance(attrib, Argument):
                args[name] = attrib

            if isinstance(attrib, Attribute):
                attributes[name] = attrib
                slots.append(name)
            else:
                attribs[name] = attrib

        # merge the annotations with the parent annotations
        signature = Signature.merge(*signatures, **args)
        argnames = tuple(signature.parameters.keys())

        attribs.update(
            __argnames__=argnames,
            __attributes__=attributes,
            __match_args__=argnames,
            __signature__=signature,
            __slots__=tuple(slots),
        )
        return super().__new__(metacls, clsname, bases, attribs, **kwargs)


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
        for name, field in self.__attributes__.items():
            if isinstance(field, Initialized):
                object.__setattr__(self, name, field.initialize(self))

    def __setattr__(self, name, value):
        if field := self.__attributes__.get(name):
            value = field.validate(value, this=self)
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
    def __setattr__(self, name: str, _: Any) -> None:
        raise TypeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )


class Singleton(Base):

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


# TODO(kszucs): consider to move it somewhere else, like ibis/expr/core.py
class Concrete(Immutable, Comparable, Annotable):

    __slots__ = ("__args__", "__precomputed_hash__")

    def __post_init__(self):
        # optimizations to store frequently accessed attributes
        argvalues = tuple(getattr(self, name) for name in self.__argnames__)
        # precompute the hash value to avoid repeating expensive hashing
        hashvalue = hash((self.__class__, argvalues))
        object.__setattr__(self, "__args__", argvalues)
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
