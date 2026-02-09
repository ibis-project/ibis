from __future__ import annotations

import inspect
from functools import partial

from singlestoredb.mysql.constants import FIELD_TYPE, FLAG

import ibis.expr.datatypes as dt

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

_type_codes = {v: k for k, v in inspect.getmembers(FIELD_TYPE) if not k.startswith("_")}

# Add float16 vector type codes if not present (for SDK version compatibility)
_type_codes.setdefault(3007, "FLOAT16_VECTOR")
_type_codes.setdefault(2007, "FLOAT16_VECTOR_JSON")


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


def _type_from_cursor_info(
    *,
    flags,
    type_code,
    field_length,
    scale,
    multi_byte_maximum_length,
    precision=None,
    charset=None,
) -> dt.DataType:
    """Construct an ibis type from SingleStoreDB field metadata.

    SingleStoreDB uses the MySQL protocol, so this closely follows
    the MySQL implementation with SingleStoreDB-specific considerations.

    Note: HTTP protocol provides limited metadata compared to MySQL protocol.
    Some types (BIT, DECIMAL, VARCHAR with specific lengths) may have reduced
    precision in schema detection when using HTTP protocol.
    """
    flags = _FieldFlags(flags)
    typename = _type_codes.get(type_code)

    if typename is None:
        raise NotImplementedError(
            f"SingleStoreDB type code {type_code:d} is not supported"
        )

    if typename in ("DECIMAL", "NEWDECIMAL"):
        # Both MySQL and HTTP protocols provide precision and scale explicitly in cursor description
        if precision is not None and scale is not None:
            typ = partial(_type_mapping[typename], precision=precision, scale=scale)
        elif scale is not None:
            typ = partial(_type_mapping[typename], scale=scale)
        else:
            typ = _type_mapping[typename]  # Generic Decimal without precision/scale
    elif typename == "BIT":
        # HTTP protocol may not provide field_length or precision
        # This is a known limitation - HTTP protocol lacks detailed type metadata
        if field_length is None or field_length == 0:
            if precision is not None and precision > 0:
                # For BIT type, HTTP protocol may store bit length in precision
                field_length = precision
            else:
                # HTTP protocol limitation: default to BIT(64) when no info available
                # This may not match the actual column definition but is the best we can do
                field_length = 64

        if field_length <= 8:
            typ = dt.int8
        elif field_length <= 16:
            typ = dt.int16
        elif field_length <= 32:
            typ = dt.int32
        elif field_length <= 64:
            typ = dt.int64
        else:
            raise AssertionError(f"invalid field length for BIT type: {field_length}")
    elif typename == "TINY" and field_length == 1:
        # TINYINT(1) is commonly used as BOOLEAN in MySQL/SingleStoreDB
        # Note: SingleStoreDB BOOLEAN columns show field_length=4 at cursor level,
        # making them indistinguishable from TINYINT. The DESCRIBE-based schema
        # detection (via to_ibis method) can properly distinguish these types.
        typ = dt.Boolean
    elif flags.is_set:
        # Sets are limited to strings in SingleStoreDB
        typ = dt.Array(dt.string)
    elif type_code in TEXT_TYPES:
        # Check charset 63 (binary charset) to distinguish binary from text
        # Both MySQL and HTTP protocols provide this info at cursor index 8
        is_binary_type = flags.is_binary or (charset == 63)

        if is_binary_type:
            typ = dt.Binary
        # For TEXT, MEDIUMTEXT, LONGTEXT (BLOB, MEDIUM_BLOB, LONG_BLOB)
        # don't include length as they are variable-length text types
        elif typename in ("BLOB", "MEDIUM_BLOB", "LONG_BLOB"):
            typ = dt.String  # No length parameter for unlimited text types
        # For VARCHAR, CHAR, etc. include the length if available
        elif field_length is not None:
            typ = partial(dt.String, length=field_length // multi_byte_maximum_length)
        else:
            # HTTP protocol: field_length is None, use String without length
            # This is a known limitation of HTTP protocol
            typ = dt.String
    elif flags.is_timestamp or typename == "TIMESTAMP":
        # SingleStoreDB timestamps - note timezone handling
        # SingleStoreDB stores timestamps in UTC by default in columnstore tables
        typ = partial(dt.Timestamp, timezone="UTC", scale=scale or None)
    elif typename == "DATETIME":
        # DATETIME doesn't have timezone info in SingleStoreDB
        # HTTP protocol: use precision from col_info[4] when scale is None
        datetime_scale = scale if scale is not None else precision
        typ = partial(dt.Timestamp, scale=datetime_scale or None)
    elif typename == "JSON":
        # SingleStoreDB has enhanced JSON support with columnstore optimizations
        typ = dt.JSON
    elif typename == "GEOGRAPHY":
        # SingleStoreDB extended geospatial type
        typ = dt.Geometry
    else:
        typ = _type_mapping[typename]
        # Only apply unsigned logic to actual type classes, not partials
        if (
            hasattr(typ, "__mro__")
            and issubclass(typ, dt.SignedInteger)
            and flags.is_unsigned
        ):
            typ = getattr(dt, f"U{typ.__name__}")

    # Projection columns are always nullable
    return typ(nullable=True)


def _decimal_length_to_precision(*, length: int, scale: int, is_unsigned: bool) -> int:
    """Calculate decimal precision from length and scale.

    Ported from MySQL's my_decimal.h:my_decimal_length_to_precision
    """
    return length - (scale > 0) - (not (is_unsigned or not length))


_type_mapping = {
    # Basic numeric types
    "DECIMAL": dt.Decimal,
    "TINY": dt.Int8,
    "SHORT": dt.Int16,
    "LONG": dt.Int32,
    "FLOAT": dt.Float32,
    "DOUBLE": dt.Float64,
    "LONGLONG": dt.Int64,
    "INT24": dt.Int32,
    "NEWDECIMAL": dt.Decimal,
    # String types
    "VARCHAR": dt.String,
    "VAR_STRING": dt.String,
    "STRING": dt.String,
    "ENUM": dt.String,
    # Temporal types
    "DATE": dt.Date,
    "TIME": dt.Time,
    "DATETIME": dt.Timestamp,
    "YEAR": dt.UInt8,
    # Binary types
    "TINY_BLOB": dt.Binary,
    "MEDIUM_BLOB": dt.Binary,
    "LONG_BLOB": dt.Binary,
    "BLOB": dt.Binary,
    # Special types
    "JSON": dt.JSON,
    "GEOMETRY": dt.Geometry,
    "NULL": dt.Null,
    # Collection types
    "SET": partial(dt.Array, dt.String),
    # SingleStoreDB-specific types
    "BSON": dt.JSON,
    # Vector types for machine learning and AI workloads
    "VECTOR": partial(dt.Array, dt.Float32),  # General vector type
    "FLOAT32_VECTOR": partial(dt.Array, dt.Float32),
    "FLOAT64_VECTOR": partial(dt.Array, dt.Float64),
    "INT8_VECTOR": partial(dt.Array, dt.Int8),
    "INT16_VECTOR": partial(dt.Array, dt.Int16),
    "INT32_VECTOR": partial(dt.Array, dt.Int32),
    "INT64_VECTOR": partial(dt.Array, dt.Int64),
    "FLOAT16_VECTOR": partial(dt.Array, dt.Float16),
    # Vector JSON types (stored as JSON with vector semantics)
    "FLOAT32_VECTOR_JSON": partial(dt.Array, dt.Float32),
    "FLOAT64_VECTOR_JSON": partial(dt.Array, dt.Float64),
    "INT8_VECTOR_JSON": partial(dt.Array, dt.Int8),
    "INT16_VECTOR_JSON": partial(dt.Array, dt.Int16),
    "INT32_VECTOR_JSON": partial(dt.Array, dt.Int32),
    "INT64_VECTOR_JSON": partial(dt.Array, dt.Int64),
    "FLOAT16_VECTOR_JSON": partial(dt.Array, dt.Float16),
    # Extended types (SingleStoreDB-specific extensions)
    "GEOGRAPHY": dt.Geometry,  # Enhanced geospatial support
}
