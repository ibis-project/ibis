from __future__ import annotations

import re
import sys
from abc import abstractmethod
from itertools import zip_longest
from types import ModuleType  # noqa: F401
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,  # noqa: F401
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from typing import get_type_hints as _get_type_hints

from ibis.common.bases import Abstract
from ibis.common.caching import memoize

if TYPE_CHECKING:
    from typing_extensions import Self

if sys.version_info >= (3, 10):
    from types import UnionType
    from typing import TypeAlias

    # Keep this alias in sync with unittest.case._ClassInfo
    _ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]
else:
    from typing_extensions import TypeAlias  # noqa: TCH002

    UnionType = object()
    _ClassInfo: TypeAlias = Union[type, tuple["_ClassInfo", ...]]


T = TypeVar("T")
U = TypeVar("U")

Namespace = dict[str, Any]
VarTuple = tuple[T, ...]


@memoize
def get_type_hints(
    obj: Any,
    include_extras: bool = True,
    include_properties: bool = False,
) -> dict[str, Any]:
    """Get type hints for a callable or class.

    Extension of typing.get_type_hints that supports getting type hints for
    class properties.

    Parameters
    ----------
    obj
        Callable or class to get type hints for.
    include_extras
        Whether to include extra type hints such as Annotated.
    include_properties
        Whether to include type hints for class properties.

    Returns
    -------
    Mapping of parameter or attribute name to type hint.
    """
    try:
        hints = _get_type_hints(obj, include_extras=include_extras)
    except TypeError:
        return {}

    if include_properties:
        for name in dir(obj):
            attr = getattr(obj, name)
            if isinstance(attr, property):
                annots = _get_type_hints(attr.fget, include_extras=include_extras)
                if return_annot := annots.get("return"):
                    hints[name] = return_annot

    return hints


@memoize
def get_type_params(obj: Any) -> dict[str, type]:
    """Get type parameters for a generic class.

    Parameters
    ----------
    obj
        Generic class to get type parameters for.

    Returns
    -------
    Mapping of type parameter name to type.

    Examples
    --------
    >>> from typing import Dict, List
    >>>
    >>> class MyList(List[T]):
    ...     ...
    ...
    >>>
    >>> get_type_params(MyList[int])
    {'T': <class 'int'>}
    >>>
    >>> class MyDict(Dict[T, U]):
    ...     ...
    ...
    >>>
    >>> get_type_params(MyDict[int, str])
    {'T': <class 'int'>, 'U': <class 'str'>}
    """
    args = get_args(obj)
    origin = get_origin(obj) or obj
    bases = getattr(origin, "__orig_bases__", ())
    params = getattr(origin, "__parameters__", ())

    result = {}
    for base in bases:
        result.update(get_type_params(base))

    param_names = (p.__name__ for p in params)
    result.update(zip(param_names, args))

    return result


@memoize
def get_bound_typevars(obj: Any) -> dict[TypeVar, tuple[str, type]]:
    """Get type variables bound to concrete types for a generic class.

    Parameters
    ----------
    obj
        Generic class to get type variables for.

    Returns
    -------
    Mapping of type variable to attribute name and type.

    Examples
    --------
    >>> class MyStruct(Generic[T, U]):
    ...     a: T
    ...     b: U
    ...
    >>> get_bound_typevars(MyStruct[int, str])
    {~T: ('a', <class 'int'>), ~U: ('b', <class 'str'>)}
    >>>
    >>> class MyStruct(Generic[T, U]):
    ...     a: T
    ...
    ...     @property
    ...     def myprop(self) -> U:
    ...         ...
    ...
    >>> get_bound_typevars(MyStruct[float, bytes])
    {~T: ('a', <class 'float'>), ~U: ('myprop', <class 'bytes'>)}
    """
    origin = get_origin(obj) or obj
    hints = get_type_hints(origin, include_properties=True)
    params = get_type_params(obj)

    result = {}
    for attr, typ in hints.items():
        if isinstance(typ, TypeVar):
            result[typ] = (attr, params[typ.__name__])
    return result


def evaluate_annotations(
    annots: dict[str, str],
    module_name: str,
    class_name: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate type annotations that are strings.

    Parameters
    ----------
    annots
        Type annotations to evaluate.
    module_name
        The name of the module that the annotations are defined in, hence
        providing global scope.
    class_name
        The name of the class that the annotations are defined in, hence
        providing Self type.

    Returns
    -------
    Actual type hints.

    Examples
    --------
    >>> annots = {"a": "dict[str, float]", "b": "int"}
    >>> evaluate_annotations(annots, __name__)
    {'a': dict[str, float], 'b': <class 'int'>}
    """
    module = sys.modules.get(module_name, None)
    globalns = getattr(module, "__dict__", None)
    if class_name is None:
        localns = None
    else:
        localns = dict(Self=f"{module_name}.{class_name}")
    return {
        k: eval(v, globalns, localns) if isinstance(v, str) else v  # noqa: PGH001
        for k, v in annots.items()
    }


def format_typehint(typ: Any) -> str:
    if isinstance(typ, type):
        return typ.__name__
    elif isinstance(typ, TypeVar):
        if typ.__bound__ is None:
            return str(typ)
        else:
            return format_typehint(typ.__bound__)
    else:
        # remove the module name from the typehint, including generics
        return re.sub(r"(\w+\.)+", "", str(typ))


class DefaultTypeVars:
    """Enable using default type variables in generic classes (PEP-0696)."""

    __slots__ = ()

    def __class_getitem__(cls, params):
        params = params if isinstance(params, tuple) else (params,)
        pairs = zip_longest(params, cls.__parameters__)
        params = tuple(p.__default__ if t is None else t for t, p in pairs)
        return super().__class_getitem__(params)


class Sentinel(type):
    """Create type-annotable unique objects."""

    def __new__(cls, name, bases, namespace, **kwargs):
        if bases:
            raise TypeError("Sentinels cannot be subclassed")
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("Sentinels are not constructible")


class CoercionError(Exception):
    ...


class Coercible(Abstract):
    """Protocol for defining coercible types.

    Coercible types define a special ``__coerce__`` method that accepts an object
    with an instance of the type. Used in conjunction with the ``coerced_to``
    pattern to coerce arguments to a specific type.
    """

    @classmethod
    @abstractmethod
    def __coerce__(cls, value: Any, **kwargs: Any) -> Self:
        ...
