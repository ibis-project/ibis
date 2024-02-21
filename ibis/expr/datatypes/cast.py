from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

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
def castable(source: dt.DataType, target: dt.DataType, value: Any = None) -> bool:
    """Return whether source ir type is implicitly castable to target."""
    from ibis.expr.datatypes.value import normalizable

    if source == target:
        return True
    elif source.is_null():
        # The null type is castable to any type, even if the target type is *not*
        # nullable.
        #
        # We handle the promotion of `null + !T -> T` at the `castable` call site.
        #
        # It might be possible to build a system with a single function that tries
        # to promote types and use the exception to indicate castability, but that
        # is a deeper refactor to be tackled later.
        #
        # See https://github.com/ibis-project/ibis/issues/2891 for the bug report
        return True
    elif target.is_boolean():
        if source.is_boolean():
            return True
        elif source.is_integer():
            return value in (0, 1)
        else:
            return False
    elif target.is_integer():
        # TODO(kszucs): ideally unsigned to signed shouldn't be allowed but that
        # breaks the integral promotion rule logic in rules.py
        if source.is_integer():
            if value is not None:
                return normalizable(target, value)
            else:
                return source.nbytes <= target.nbytes
        else:
            return False
    elif target.is_floating():
        if source.is_floating():
            return source.nbytes <= target.nbytes
        else:
            return source.is_integer()
    elif target.is_decimal():
        if source.is_decimal():
            downcast_precision = (
                source.precision is not None
                and target.precision is not None
                and source.precision < target.precision
            )
            downcast_scale = (
                source.scale is not None
                and target.scale is not None
                and source.scale < target.scale
            )
            return not (downcast_precision or downcast_scale)
        else:
            return source.is_numeric()
    elif target.is_string():
        return source.is_string() or source.is_uuid()
    elif target.is_uuid():
        return source.is_uuid() or source.is_string()
    elif target.is_date() or target.is_timestamp():
        if source.is_string():
            return value is not None and normalizable(target, value)
        else:
            return source.is_timestamp() or source.is_date()
    elif target.is_interval():
        if source.is_interval():
            return source.unit == target.unit
        else:
            return source.is_integer()
    elif target.is_time():
        if source.is_string():
            return value is not None and normalizable(target, value)
        else:
            return source.is_time()
    elif target.is_json():
        return (
            source.is_json()
            or source.is_string()
            or source.is_floating()
            or source.is_integer()
        )
    elif target.is_array():
        return source.is_array() and castable(source.value_type, target.value_type)
    elif target.is_map():
        return (
            source.is_map()
            and castable(source.key_type, target.key_type)
            and castable(source.value_type, target.value_type)
        )
    elif target.is_struct():
        return source.is_struct() and all(
            castable(source[field], target[field]) for field in target.names
        )
    elif target.is_geospatial():
        return source.is_geospatial() or source.is_array()
    else:
        return isinstance(target, source.__class__)


@public
def higher_precedence(left: dt.DataType, right: dt.DataType) -> dt.DataType:
    nullable = left.nullable or right.nullable

    if left.castable(right):
        return right.copy(nullable=nullable)
    elif right.castable(left):
        return left.copy(nullable=nullable)
    else:
        raise IbisTypeError(
            f"Cannot compute precedence for `{left}` and `{right}` types"
        )


@public
def highest_precedence(dtypes: Iterator[dt.DataType]) -> dt.DataType:
    """Compute the highest precedence of `dtypes`."""
    if collected := list(dtypes):
        return functools.reduce(higher_precedence, collected)
    else:
        return dt.null
