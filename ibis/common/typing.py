from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Iterable,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)

import toolz

# TODO(kszucs): try to use inspect.get_annotations() backport instead

if sys.version_info >= (3, 9):

    @toolz.memoize
    def evaluate_typehint(hint, module_name=None) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            if module_name is None:
                globalns = {}
            else:
                globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals(), frozenset())
        else:
            return hint

else:

    @toolz.memoize
    def evaluate_typehint(hint, module_name) -> Any:
        if isinstance(hint, str):
            hint = ForwardRef(hint)
        if isinstance(hint, ForwardRef):
            if module_name is None:
                globalns = {}
            else:
                globalns = sys.modules[module_name].__dict__
            return hint._evaluate(globalns, locals())
        else:
            return hint


if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    import ibis.expr.schema as sch

    SupportsSchema = TypeVar(
        "SupportsSchema",
        Iterable[Tuple[str, Union[str, dt.DataType]]],
        Mapping[str, Union[str, dt.DataType]],
        sch.Schema,
    )
