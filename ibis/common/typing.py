from __future__ import annotations

import inspect
import sys
from itertools import zip_longest
from types import UnionType
from typing import Any, Optional, TypeAlias, TypeVar

# Keep this alias in sync with unittest.case._ClassInfo
_ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]


T = TypeVar("T")
U = TypeVar("U")

Namespace = dict[str, Any]
VarTuple = tuple[T, ...]


def evaluate_annotations(
    annots: dict[str, str],
    module_name: str,
    class_name: Optional[str] = None,
    best_effort: bool = False,
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
    best_effort
        Whether to ignore errors when evaluating type annotations.

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

    result = {}
    for k, v in annots.items():
        if isinstance(v, str):
            try:
                v = eval(v, globalns, localns)  # noqa: S307
            except NameError:
                if not best_effort:
                    raise
        result[k] = v

    return result


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


def get_defining_frame(obj):
    """Locate the outermost frame where `obj` is defined."""
    for frame_info in inspect.stack()[::-1]:
        for var in frame_info.frame.f_locals.values():
            if obj is var:
                return frame_info.frame
    raise ValueError(f"No defining frame found for {obj}")


def get_defining_scope(obj, types=None):
    """Get variables in the scope where `expr` is first defined."""
    frame = get_defining_frame(obj)
    scope = {**frame.f_globals, **frame.f_locals}
    if types is not None:
        scope = {k: v for k, v in scope.items() if isinstance(v, types)}
    return scope
