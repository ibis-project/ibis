from __future__ import annotations

import collections.abc
from abc import abstractmethod
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

from ibis.common.caching import WeakCache

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self


class AbstractMeta(type):
    """Base metaclass for many of the ibis core classes.

    Enforce the subclasses to define a `__slots__` attribute and provide a
    `__create__` classmethod to change the instantiation behavior of the class.

    Support abstract methods without extending `abc.ABCMeta`. While it provides
    a reduced feature set compared to `abc.ABCMeta` (no way to register virtual
    subclasses) but avoids expensive instance checks by enforcing explicit
    subclassing.
    """

    __slots__ = ()

    def __new__(metacls, clsname, bases, dct, **kwargs):
        # enforce slot definitions
        dct.setdefault("__slots__", ())

        # construct the class object
        cls = super().__new__(metacls, clsname, bases, dct, **kwargs)

        # calculate abstract methods existing in the class
        abstracts = {
            name
            for name, value in dct.items()
            if getattr(value, "__isabstractmethod__", False)
        }
        for parent in bases:
            for name in getattr(parent, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)

        # set the abstract methods for the class
        cls.__abstractmethods__ = frozenset(abstracts)

        return cls

    def __call__(cls, *args, **kwargs):
        """Create a new instance of the class.

        The subclass may override the `__create__` classmethod to change the
        instantiation behavior. This is similar to overriding the `__new__`
        method, but without conditionally calling the `__init__` based on the
        return type.

        Parameters
        ----------
        args : tuple
            Positional arguments eventually passed to the `__init__` method.
        kwargs : dict
            Keyword arguments eventually passed to the `__init__` method.

        Returns
        -------
        The newly created instance of the class. No extra initialization
        """
        return cls.__create__(*args, **kwargs)


class Abstract(metaclass=AbstractMeta):
    """Base class for many of the ibis core classes, see `AbstractMeta`."""

    __slots__ = ("__weakref__",)
    __create__ = classmethod(type.__call__)  # type: ignore


class Immutable(Abstract):
    """Prohibit attribute assignment on the instance."""

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __setattr__(self, name: str, _: Any) -> None:
        raise AttributeError(
            f"Attribute {name!r} cannot be assigned to immutable instance of "
            f"type {type(self)}"
        )


class Singleton(Abstract):
    """Cache instances of the class based on instantiation arguments."""

    __instances__: Mapping[Any, Self] = WeakValueDictionary()

    @classmethod
    def __create__(cls, *args, **kwargs):
        key = (cls, args, tuple(kwargs.items()))
        try:
            return cls.__instances__[key]
        except KeyError:
            instance = super().__create__(*args, **kwargs)
            cls.__instances__[key] = instance
            return instance


class Final(Abstract):
    """Prohibit subclassing."""

    def __init_subclass__(cls, **kwargs):
        cls.__init_subclass__ = cls.__prohibit_inheritance__

    @classmethod
    def __prohibit_inheritance__(cls, **kwargs):
        raise TypeError(f"Cannot inherit from final class {cls}")


@collections.abc.Hashable.register
class Hashable(Abstract):
    @abstractmethod
    def __hash__(self) -> int:
        ...


class Comparable(Abstract):
    """Enable quick equality comparisons.

    The subclasses must implement the `__equals__` method that returns a boolean
    value indicating whether the two instances are equal. This method is called
    only if the two instances are of the same type and the result is cached for
    future comparisons.

    Since the class holds a global cache of comparison results, it is important
    to make sure that the instances are not kept alive longer than necessary.
    This is done automatically by using weak references for the compared objects.
    """

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


class Slotted(Abstract):
    """A lightweight alternative to `ibis.common.grounds.Annotable`.

    The class is mostly used to reduce boilerplate code.
    """

    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in self.__slots__)

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __repr__(self):
        fields = {k: getattr(self, k) for k in self.__slots__}
        fieldstring = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fieldstring})"

    def __rich_repr__(self):
        for name in self.__slots__:
            yield name, getattr(self, name)


class FrozenSlotted(Slotted, Immutable, Hashable):
    """A lightweight alternative to `ibis.common.grounds.Concrete`.

    This class is used to create immutable dataclasses with slots and a precomputed
    hash value for quicker dictionary lookups.
    """

    __slots__ = ("__precomputed_hash__",)
    __precomputed_hash__: int

    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        hashvalue = hash(tuple(kwargs.values()))
        object.__setattr__(self, "__precomputed_hash__", hashvalue)

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        hashvalue = hash(tuple(state.values()))
        object.__setattr__(self, "__precomputed_hash__", hashvalue)

    def __hash__(self) -> int:
        return self.__precomputed_hash__
