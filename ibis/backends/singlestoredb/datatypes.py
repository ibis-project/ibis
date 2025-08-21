from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import ibis.expr.datatypes as dt
from ibis.backends.sql.datatypes import SqlglotType

if TYPE_CHECKING:
    try:
        from MySQLdb.constants import FIELD_TYPE, FLAG
    except ImportError:
        # Fallback for when MySQLdb is not available
        FIELD_TYPE = None
        FLAG = None

# SingleStoreDB uses the MySQL protocol, so we can reuse MySQL type constants
# when available, otherwise define our own minimal set
try:
    from MySQLdb.constants import FIELD_TYPE, FLAG

    TEXT_TYPES = (
        FIELD_TYPE.BIT,
        FIELD_TYPE.BLOB,
        FIELD_TYPE.LONG_BLOB,
        FIELD_TYPE.MEDIUM_BLOB,
        FIELD_TYPE.STRING,
        FIELD_TYPE.TINY_BLOB,
        FIELD_TYPE.VAR_STRING,
        FIELD_TYPE.VARCHAR,
        FIELD_TYPE.GEOMETRY,
    )

    _type_codes = {
        v: k for k, v in inspect.getmembers(FIELD_TYPE) if not k.startswith("_")
    }

    class _FieldFlags:
        """Flags used to disambiguate field types for SingleStoreDB."""

        __slots__ = ("value",)

        def __init__(self, value: int) -> None:
            self.value = value

        @property
        def is_unsigned(self) -> bool:
            return (FLAG.UNSIGNED & self.value) != 0

        @property
        def is_timestamp(self) -> bool:
            return (FLAG.TIMESTAMP & self.value) != 0

        @property
        def is_set(self) -> bool:
            return (FLAG.SET & self.value) != 0

        @property
        def is_num(self) -> bool:
            return (FLAG.NUM & self.value) != 0

        @property
        def is_binary(self) -> bool:
            return (FLAG.BINARY & self.value) != 0

except ImportError:
    # Fallback when MySQLdb is not available
    TEXT_TYPES = (0, 249, 250, 251, 252, 253, 254, 255)  # Basic type codes
    _type_codes = {
        0: "DECIMAL",
        1: "TINY",
        2: "SHORT",
        3: "LONG",
        4: "FLOAT",
        5: "DOUBLE",
        6: "NULL",
        7: "TIMESTAMP",
        8: "LONGLONG",
        9: "INT24",
        10: "DATE",
        11: "TIME",
        12: "DATETIME",
        13: "YEAR",
        15: "VARCHAR",
        16: "BIT",
        245: "JSON",
        246: "NEWDECIMAL",
        247: "ENUM",
        248: "SET",
        249: "TINY_BLOB",
        250: "MEDIUM_BLOB",
        251: "LONG_BLOB",
        252: "BLOB",
        253: "VAR_STRING",
        254: "STRING",
        255: "GEOMETRY",
    }

    class _FieldFlags:
        """Fallback field flags implementation."""

        __slots__ = ("value",)

        def __init__(self, value: int) -> None:
            self.value = value

        @property
        def is_unsigned(self) -> bool:
            return (32 & self.value) != 0  # UNSIGNED_FLAG = 32

        @property
        def is_timestamp(self) -> bool:
            return (1024 & self.value) != 0  # TIMESTAMP_FLAG = 1024

        @property
        def is_set(self) -> bool:
            return (2048 & self.value) != 0  # SET_FLAG = 2048

        @property
        def is_num(self) -> bool:
            return (32768 & self.value) != 0  # NUM_FLAG = 32768

        @property
        def is_binary(self) -> bool:
            return (128 & self.value) != 0  # BINARY_FLAG = 128


def _type_from_cursor_info(
    *, flags, type_code, field_length, scale, multi_byte_maximum_length
) -> dt.DataType:
    """Construct an ibis type from SingleStoreDB field metadata.

    SingleStoreDB uses the MySQL protocol, so this closely follows
    the MySQL implementation with SingleStoreDB-specific considerations.
    """
    flags = _FieldFlags(flags)
    typename = _type_codes.get(type_code)
    if typename is None:
        raise NotImplementedError(
            f"SingleStoreDB type code {type_code:d} is not supported"
        )

    if typename in ("DECIMAL", "NEWDECIMAL"):
        precision = _decimal_length_to_precision(
            length=field_length, scale=scale, is_unsigned=flags.is_unsigned
        )
        typ = partial(_type_mapping[typename], precision=precision, scale=scale)
    elif typename == "BIT":
        if field_length <= 8:
            typ = dt.int8
        elif field_length <= 16:
            typ = dt.int16
        elif field_length <= 32:
            typ = dt.int32
        elif field_length <= 64:
            typ = dt.int64
        else:
            raise AssertionError("invalid field length for BIT type")
    elif flags.is_set:
        # Sets are limited to strings in SingleStoreDB
        typ = dt.Array(dt.string)
    elif type_code in TEXT_TYPES:
        if flags.is_binary:
            typ = dt.Binary
        else:
            typ = partial(dt.String, length=field_length // multi_byte_maximum_length)
    elif flags.is_timestamp or typename == "TIMESTAMP":
        # SingleStoreDB timestamps - note timezone handling
        typ = partial(dt.Timestamp, timezone="UTC", scale=scale or None)
    elif typename == "DATETIME":
        typ = partial(dt.Timestamp, scale=scale or None)
    else:
        typ = _type_mapping[typename]
        if issubclass(typ, dt.SignedInteger) and flags.is_unsigned:
            typ = getattr(dt, f"U{typ.__name__}")

    # Projection columns are always nullable
    return typ(nullable=True)


def _decimal_length_to_precision(*, length: int, scale: int, is_unsigned: bool) -> int:
    """Calculate decimal precision from length and scale.

    Ported from MySQL's my_decimal.h:my_decimal_length_to_precision
    """
    return length - (scale > 0) - (not (is_unsigned or not length))


_type_mapping = {
    "DECIMAL": dt.Decimal,
    "TINY": dt.Int8,
    "SHORT": dt.Int16,
    "LONG": dt.Int32,
    "FLOAT": dt.Float32,
    "DOUBLE": dt.Float64,
    "NULL": dt.Null,
    "LONGLONG": dt.Int64,
    "INT24": dt.Int32,
    "DATE": dt.Date,
    "TIME": dt.Time,
    "DATETIME": dt.Timestamp,
    "YEAR": dt.UInt8,
    "VARCHAR": dt.String,
    "JSON": dt.JSON,
    "NEWDECIMAL": dt.Decimal,
    "ENUM": dt.String,
    "SET": partial(dt.Array, dt.string),
    "TINY_BLOB": dt.Binary,
    "MEDIUM_BLOB": dt.Binary,
    "LONG_BLOB": dt.Binary,
    "BLOB": dt.Binary,
    "VAR_STRING": dt.String,
    "STRING": dt.String,
    "GEOMETRY": dt.Geometry,
}


class SingleStoreDBType(SqlglotType):
    """SingleStoreDB data type implementation."""

    dialect = "mysql"  # SingleStoreDB uses MySQL dialect for SQLGlot

    # SingleStoreDB-specific type mappings
    # Most types are the same as MySQL due to protocol compatibility
    default_decimal_precision = 10
    default_decimal_scale = 0
    default_temporal_scale = None

    # SingleStoreDB supports these additional types beyond standard MySQL
    # These may be added in future versions
    # VECTOR - for machine learning workloads (not yet implemented)
    # GEOGRAPHY - enhanced geospatial support (maps to GEOMETRY for now)

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        """Convert SingleStoreDB type to Ibis type."""
        # For now, delegate to the parent MySQL-compatible implementation
        return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        """Convert Ibis type to SingleStoreDB type."""
        # For now, delegate to the parent MySQL-compatible implementation
        return super().from_ibis(dtype)
