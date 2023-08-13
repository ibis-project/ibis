from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from multipledispatch import Dispatcher
from public import public

import ibis.expr.datatypes.core as dt
from ibis.common.exceptions import IbisTypeError

if TYPE_CHECKING:
    from collections.abc import Iterator

castable = Dispatcher("castable")


@public
def cast(source: str | dt.DataType, target: str | dt.DataType, **kwargs) -> dt.DataType:
    """Attempts to implicitly cast from source dtype to target dtype."""
    source, target = dt.dtype(source), dt.dtype(target)

    if not castable(source, target, **kwargs):
        raise IbisTypeError(
            f"Datatype {source} cannot be implicitly casted to {target}"
        )
    return target


@public
def higher_precedence(left: dt.DataType, right: dt.DataType) -> dt.DataType:
    nullable = left.nullable or right.nullable

    if castable(left, right, upcast=True):
        return right.copy(nullable=nullable)
    elif castable(right, left, upcast=True):
        return left.copy(nullable=nullable)

    raise IbisTypeError(f"Cannot compute precedence for `{left}` and `{right}` types")


@public
def highest_precedence(dtypes: Iterator[dt.DataType]) -> dt.DataType:
    """Compute the highest precedence of `dtypes`."""
    if collected := list(dtypes):
        return functools.reduce(higher_precedence, collected)
    else:
        return dt.null


@castable.register(dt.DataType, dt.DataType)
def can_cast_subtype(source: dt.DataType, target: dt.DataType, **kwargs) -> bool:
    return isinstance(target, source.__class__)


@castable.register(dt.Integer, (dt.Floating, dt.Decimal))
@castable.register(dt.Floating, dt.Decimal)
@castable.register((dt.Date, dt.Timestamp), (dt.Date, dt.Timestamp))
@castable.register(dt.String, dt.JSON)
@castable.register(dt.JSON, dt.String)
def can_cast_any(source: dt.DataType, target: dt.DataType, **kwargs) -> bool:
    return True


@castable.register(dt.Null, dt.DataType)
def can_cast_null(source: dt.DataType, target: dt.DataType, **kwargs) -> bool:
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


@castable.register(dt.SignedInteger, dt.UnsignedInteger)
@castable.register(dt.UnsignedInteger, dt.SignedInteger)
def can_cast_to_differently_signed_integer_type(
    source: dt.Integer, target: dt.Integer, value: int | None = None, **kwargs
) -> bool:
    if value is not None:
        # TODO(kszucs): we may not need to actually check the value since the
        # literal construction now checks for bounds and doesn't use castable()
        # anymore
        return target.bounds.lower <= value <= target.bounds.upper
    else:
        return (target.bounds.upper - target.bounds.lower) >= (
            source.bounds.upper - source.bounds.lower
        )


@castable.register(dt.SignedInteger, dt.SignedInteger)
@castable.register(dt.UnsignedInteger, dt.UnsignedInteger)
def can_cast_integers(source: dt.Integer, target: dt.Integer, **kwargs) -> bool:
    return target.nbytes >= source.nbytes


@castable.register(dt.Floating, dt.Floating)
def can_cast_floats(
    source: dt.Floating, target: dt.Floating, upcast: bool = False, **kwargs
) -> bool:
    if upcast:
        return target.nbytes >= source.nbytes

    # double -> float must be allowed because
    # float literals are inferred as doubles
    return True


@castable.register(dt.Decimal, dt.Decimal)
def can_cast_decimals(source: dt.Decimal, target: dt.Decimal, **kwargs) -> bool:
    target_prec = target.precision
    source_prec = source.precision
    target_sc = target.scale
    source_sc = source.scale
    return (
        # If either sides precision and scale are both `None`, return `True`.
        target_prec is None
        and target_sc is None
        or source_prec is None
        and source_sc is None
        # Otherwise, return `True` unless we are downcasting precision or scale.
        or (
            target_prec is None
            or (source_prec is not None and target_prec >= source_prec)
        )
        and (target_sc is None or (source_sc is not None and target_sc >= source_sc))
    )


@castable.register(dt.Interval, dt.Interval)
def can_cast_intervals(source: dt.Interval, target: dt.Interval, **kwargs) -> bool:
    return source.unit == target.unit


@castable.register(dt.Integer, dt.Boolean)
def can_cast_integer_to_boolean(
    source: dt.Integer, target: dt.Boolean, value: int | None = None, **kwargs
) -> bool:
    return value is not None and 0 <= value <= 1


@castable.register(dt.Integer, dt.Interval)
def can_cast_integer_to_interval(
    source: dt.Integer, target: dt.Interval, **kwargs
) -> bool:
    return True


@castable.register(dt.String, (dt.Date, dt.Time, dt.Timestamp))
def can_cast_string_to_temporal(
    source: dt.String,
    target: dt.Date | dt.Time | dt.Timestamp,
    value: str | None = None,
    **kwargs,
) -> bool:
    import pandas as pd

    if value is None:
        return False
    try:
        pd.Timestamp(value)
    except ValueError:
        return False
    else:
        return True


@castable.register(dt.Map, dt.Map)
def can_cast_map(source, target, **kwargs):
    return castable(source.key_type, target.key_type) and castable(
        source.value_type, target.value_type
    )


@castable.register(dt.Struct, dt.Struct)
def can_cast_struct(source, target, **kwargs):
    return all(castable(source[field], target[field]) for field in target.names)


@castable.register(dt.Array, dt.Array)
def can_cast_array_or_set(source: dt.Array, target: dt.Array, **kwargs) -> bool:
    return castable(source.value_type, target.value_type)


@castable.register(dt.JSON, dt.JSON)
@castable.register(dt.Floating, dt.JSON)
@castable.register(dt.Integer, dt.JSON)
def can_cast_json(source, target, **kwargs):
    return True


@castable.register(dt.Array, dt.GeoSpatial)
def can_cast_geospatial(source, target, **kwargs):
    return True


@castable.register(dt.UUID, dt.UUID)
@castable.register(dt.UUID, dt.String)
@castable.register(dt.String, dt.UUID)
@castable.register(dt.MACADDR, dt.MACADDR)
@castable.register(dt.INET, dt.INET)
def can_cast_special_string(source, target, **kwargs):
    return True


public(castable=castable)
