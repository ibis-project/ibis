from __future__ import annotations

import contextlib
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import (
    Any,
    ClassVar,
    Mapping,
    Tuple,
    Union,
    get_origin,
)
from weakref import WeakValueDictionary

from typing_extensions import Self, dataclass_transform

from ibis.common.annotations import (
    EMPTY,
    Annotation,
    Argument,
    Attribute,
    Signature,
    attribute,
)
from ibis.common.caching import WeakCache
from ibis.common.collections import FrozenDict
from ibis.common.patterns import Validator
from ibis.common.typing import evaluate_annotations


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
    __create__ = classmethod(type.__call__)  # type: ignore


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
        module = dct.get('__module__')
        qualname = dct.get('__qualname__') or clsname
        annotations = dct.get('__annotations__', {})

        # TODO(kszucs): pass dct as localns to evaluate_annotations
        typehints = evaluate_annotations(annotations, module)
        for name, typehint in typehints.items():
            if get_origin(typehint) is ClassVar:
                continue
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
            __module__=module,
            __qualname__=qualname,
            __argnames__=argnames,
            __attributes__=attributes,
            __match_args__=argnames,
            __signature__=signature,
            __slots__=tuple(slots),
        )
        return super().__new__(metacls, clsname, bases, namespace, **kwargs)

    def __or__(self, other):
        # required to support `dt.Numeric | dt.Floating` annotation for python<3.10
        return Union[self, other]


@dataclass_transform()
class Annotable(Base, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    __argnames__: ClassVar[Tuple[str, ...]]
    __attributes__: ClassVar[FrozenDict[str, Annotation]]
    __match_args__: ClassVar[Tuple[str, ...]]
    __signature__: ClassVar[Signature]

    @classmethod
    def __create__(cls, *args: Any, **kwargs: Any) -> Self:
        # construct the instance by passing the validated keyword arguments
        kwargs = cls.__signature__.validate(*args, **kwargs)
        return super().__create__(**kwargs)

    @classmethod
    def __recreate__(cls, kwargs: Any) -> Self:
        # bypass signature binding by requiring keyword arguments only
        kwargs = cls.__signature__.validate_nobind(**kwargs)
        return super().__create__(**kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
            value = field.validate(value, self)
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
    def __args__(self) -> Tuple[Any, ...]:
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
    __instances__: Mapping[Any, Self] = WeakValueDictionary()

    @classmethod
    def __create__(cls, *args, **kwargs):
        key = (cls, args, FrozenDict(kwargs))
        try:
            return cls.__instances__[key]
        except KeyError:
            instance = super().__create__(*args, **kwargs)
            cls.__instances__[key] = instance
            return instance


class Final(Base):
    def __init_subclass__(cls, **kwargs):
        cls.__init_subclass__ = cls.__prohibit_inheritance__

    @classmethod
    def __prohibit_inheritance__(cls, **kwargs):
        raise TypeError(f"Cannot inherit from final class {cls}")


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
    def __precomputed_hash__(self) -> int:
        return hash((self.__class__, self.__args__))

    def __reduce__(self):
        # assuming immutability and idempotency of the __init__ method, we can
        # reconstruct the instance from the arguments without additional attributes
        state = dict(zip(self.__argnames__, self.__args__))
        return (self.__recreate__, (state,))

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __equals__(self, other) -> bool:
        return self.__args__ == other.__args__

    @property
    def args(self):
        return self.__args__

    @property
    def argnames(self) -> Tuple[str, ...]:
        return self.__argnames__

    def copy(self, **overrides) -> Self:
        kwargs = dict(zip(self.__argnames__, self.__args__))
        kwargs.update(overrides)
        return self.__recreate__(kwargs)
