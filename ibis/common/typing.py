from __future__ import annotations

import sys
from itertools import zip_longest
from typing import (
    Any,
    Dict,
    Generic,  # noqa: F401
    Optional,
    Tuple,
    TypeVar,
    get_args,
    get_origin,
)

from typing_extensions import get_type_hints as _get_type_hints

from ibis.common.caching import memoize

try:
    from types import UnionType
except ImportError:
    UnionType = object()


T = TypeVar("T")
U = TypeVar("U")

Namespace = Dict[str, Any]
VarTuple = Tuple[T, ...]


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
    dict[str, Any]
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
def get_type_params(obj: Any) -> dict[str, Any]:
    """Get type parameters for a generic class.

    Parameters
    ----------
    obj
        Generic class to get type parameters for.

    Returns
    -------
    dict[str, Any]
        Mapping of type parameter name to type.

    Examples
    --------
    >>> from typing import Dict, List
    >>>
    >>> class MyList(List[T]): ...
    >>>
    >>> get_type_params(MyList[int])
    {'T': <class 'int'>}
    >>>
    >>> class MyDict(Dict[T, U]): ...
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
def get_bound_typevars(obj: Any) -> dict[str, Any]:
    """Get type variables bound to concrete types for a generic class.

    Parameters
    ----------
    obj
        Generic class to get type variables for.

    Returns
    -------
    dict[str, Any]
        Mapping of type variable name to type.

    Examples
    --------
    >>> class MyStruct(Generic[T, U]):
    ...    a: T
    ...    b: U
    ...
    >>> get_bound_typevars(MyStruct[int, str])
    {~T: ('a', <class 'int'>), ~U: ('b', <class 'str'>)}
    >>>
    >>> class MyStruct(Generic[T, U]):
    ...    a: T
    ...
    ...    @property
    ...    def myprop(self) -> U:
    ...        ...
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
    annots: dict[str, str], module_name: str, localns: Optional[Namespace] = None
) -> dict[str, Any]:
    """Evaluate type annotations that are strings.

    Parameters
    ----------
    annots
        Type annotations to evaluate.
    module_name
        The name of the module that the annotations are defined in, hence
        providing global scope.
    localns
        The local namespace to use for evaluation.

    Returns
    -------
    dict[str, Any]
        Actual type hints.

    Examples
    --------
    >>> annots = {'a': 'Dict[str, float]', 'b': 'int'}
    >>> evaluate_annotations(annots, __name__)
    {'a': typing.Dict[str, float], 'b': <class 'int'>}
    """
    module = sys.modules.get(module_name, None)
    globalns = getattr(module, '__dict__', None)
    return {
        k: eval(v, globalns, localns) if isinstance(v, str) else v  # noqa: PGH001
        for k, v in annots.items()
    }


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
