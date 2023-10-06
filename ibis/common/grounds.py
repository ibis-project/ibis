from __future__ import annotations

import contextlib
from copy import copy
from typing import (
    Any,
    ClassVar,
    Union,
    get_origin,
)

from typing_extensions import Self, dataclass_transform

from ibis.common.annotations import (
    Annotation,
    Argument,
    Attribute,
    Signature,
)
from ibis.common.bases import (  # noqa: F401
    Abstract,
    AbstractMeta,
    Comparable,
    Final,
    Immutable,
    Singleton,
)
from ibis.common.collections import FrozenDict  # noqa: TCH001
from ibis.common.patterns import Pattern
from ibis.common.typing import evaluate_annotations


class AnnotableMeta(AbstractMeta):
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

        # collection type annotations and convert them to patterns
        module = dct.get("__module__")
        qualname = dct.get("__qualname__") or clsname
        annotations = dct.get("__annotations__", {})

        # TODO(kszucs): pass dct as localns to evaluate_annotations
        typehints = evaluate_annotations(annotations, module, clsname)
        for name, typehint in typehints.items():
            if get_origin(typehint) is ClassVar:
                continue
            pattern = Pattern.from_typehint(typehint)
            if name in dct:
                dct[name] = Argument(pattern, default=dct[name], typehint=typehint)
            else:
                dct[name] = Argument(pattern, typehint=typehint)

        # collect the newly defined annotations
        slots = list(dct.pop("__slots__", []))
        namespace, arguments = {}, {}
        for name, attrib in dct.items():
            if isinstance(attrib, Pattern):
                arguments[name] = Argument(attrib)
                slots.append(name)
            elif isinstance(attrib, Argument):
                arguments[name] = attrib
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
class Annotable(Abstract, metaclass=AnnotableMeta):
    """Base class for objects with custom validation rules."""

    __signature__: ClassVar[Signature]
    """Signature of the class, containing the Argument annotations."""

    __attributes__: ClassVar[FrozenDict[str, Annotation]]
    """Mapping of the Attribute annotations."""

    __argnames__: ClassVar[tuple[str, ...]]
    """Names of the arguments."""

    __match_args__: ClassVar[tuple[str, ...]]
    """Names of the arguments to be used for pattern matching."""

    @classmethod
    def __create__(cls, *args: Any, **kwargs: Any) -> Self:
        # construct the instance by passing only validated keyword arguments
        kwargs = cls.__signature__.validate(cls, args, kwargs)
        return super().__create__(**kwargs)

    @classmethod
    def __recreate__(cls, kwargs: Any) -> Self:
        # bypass signature binding by requiring keyword arguments only
        kwargs = cls.__signature__.validate_nobind(cls, kwargs)
        return super().__create__(**kwargs)

    def __init__(self, **kwargs: Any) -> None:
        # set the already validated arguments
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        # initialize the remaining attributes
        for name, field in self.__attributes__.items():
            if field.has_default():
                object.__setattr__(self, name, field.get_default(name, self))

    def __setattr__(self, name, value) -> None:
        # first try to look up the argument then the attribute
        if param := self.__signature__.parameters.get(name):
            value = param.annotation.validate(name, value, self)
        # then try to look up the attribute
        elif annot := self.__attributes__.get(name):
            value = annot.validate(name, value, self)
        return super().__setattr__(name, value)

    def __repr__(self) -> str:
        args = (f"{n}={getattr(self, n)!r}" for n in self.__argnames__)
        argstring = ", ".join(args)
        return f"{self.__class__.__name__}({argstring})"

    def __eq__(self, other) -> bool:
        # compare types
        if type(self) is not type(other):
            return NotImplemented
        # compare arguments
        if self.__args__ != other.__args__:
            return False
        # compare attributes
        for name in self.__attributes__:
            if getattr(self, name, None) != getattr(other, name, None):
                return False
        return True

    @property
    def __args__(self) -> tuple[Any, ...]:
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


class Concrete(Immutable, Comparable, Annotable):
    """Opinionated base class for immutable data classes."""

    __slots__ = ("__args__", "__precomputed_hash__")

    def __init__(self, **kwargs: Any) -> None:
        # collect and set the arguments in a single pass
        args = []
        for name in self.__argnames__:
            value = kwargs[name]
            args.append(value)
            object.__setattr__(self, name, value)

        # precompute the hash value since the instance is immutable
        args = tuple(args)
        hashvalue = hash((self.__class__, args))
        object.__setattr__(self, "__args__", args)
        object.__setattr__(self, "__precomputed_hash__", hashvalue)

        # initialize the remaining attributes
        for name, field in self.__attributes__.items():
            if field.has_default():
                object.__setattr__(self, name, field.get_default(name, self))

    def __reduce__(self):
        # assuming immutability and idempotency of the __init__ method, we can
        # reconstruct the instance from the arguments without additional attributes
        state = dict(zip(self.__argnames__, self.__args__))
        return (self.__recreate__, (state,))

    def __hash__(self) -> int:
        return self.__precomputed_hash__

    def __equals__(self, other) -> bool:
        return hash(self) == hash(other) and self.__args__ == other.__args__

    @property
    def args(self):
        return self.__args__

    @property
    def argnames(self) -> tuple[str, ...]:
        return self.__argnames__

    def copy(self, **overrides) -> Self:
        kwargs = dict(zip(self.__argnames__, self.__args__))
        kwargs.update(overrides)
        return self.__recreate__(kwargs)
