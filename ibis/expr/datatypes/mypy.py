from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    from .core import (
        Array,
        Binary,
        Boolean,
        Date,
        Decimal,
        Float32,
        Float64,
        Int8,
        Int16,
        Int32,
        Int64,
        Interval,
        Map,
        Null,
        String,
        Struct,
        Timestamp,
    )


# Makes it possible to do exhaustive checks via mypy
_DataType: TypeAlias = Union[
    Array,
    Binary,
    Boolean,
    Date,
    Decimal,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    Int8,
    Interval,
    Map,
    Null,
    String,
    Struct,
    Timestamp,
]
