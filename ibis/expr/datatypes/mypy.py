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
else:
    Array = object
    Binary = object
    Boolean = object
    Date = object
    Decimal = object
    Float32 = object
    Float64 = object
    Int8 = object
    Int16 = object
    Int32 = object
    Int64 = object
    Interval = object
    Map = object
    Null = object
    String = object
    Struct = object
    Timestamp = object


