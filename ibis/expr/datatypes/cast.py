from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from public import public

import ibis.expr.datatypes.core as dt
from ibis.common.exceptions import IbisTypeError

if TYPE_CHECKING:
    from collections.abc import Iterator


@public
def cast(source: str | dt.DataType, target: str | dt.DataType, **kwargs) -> dt.DataType:
    """Attempts to implicitly cast from source dtype to target dtype."""
    source, target = dt.dtype(source), dt.dtype(target)

    if not source.castable(target, **kwargs):
        raise IbisTypeError(
            f"Datatype {source} cannot be implicitly casted to {target}"
        )
    return target


@public
def higher_precedence(left: dt.DataType, right: dt.DataType) -> dt.DataType:
    nullable = left.nullable or right.nullable

    if left.castable(right, upcast=True):
        return right.copy(nullable=nullable)
    elif right.castable(left, upcast=True):
        return left.copy(nullable=nullable)

    raise IbisTypeError(f"Cannot compute precedence for `{left}` and `{right}` types")


@public
def highest_precedence(dtypes: Iterator[dt.DataType]) -> dt.DataType:
    """Compute the highest precedence of `dtypes`."""
    if collected := list(dtypes):
        return functools.reduce(higher_precedence, collected)
    else:
        return dt.null
