from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import sqlglot.expressions as sge

import ibis.expr.datatypes as dt
from ibis.backends.sql.datatypes import SqlglotType

if TYPE_CHECKING:
    try:
        from singlestoredb.mysql.constants import FIELD_TYPE, FLAG
    except ImportError:
        FIELD_TYPE = None
        FLAG = None

try:
    from singlestoredb.mysql.constants import FIELD_TYPE, FLAG

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
        # SingleStoreDB-specific type codes (hypothetical values)
        256: "VECTOR",  # Vector type for ML/AI workloads
        257: "GEOGRAPHY",  # Extended geospatial support
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

    # Handle SingleStoreDB vector types that may not be in _type_codes
    if type_code in (3001, 3002, 3003, 3004, 3005, 3006):  # Vector types
        # SingleStoreDB VECTOR types - map to Binary for now
        # Could be enhanced to Array[Float32] or other appropriate types in future
        return dt.Binary(nullable=True)
    elif type_code in (2001, 2002, 2003, 2004, 2005, 2006):  # Vector JSON types
        # SingleStoreDB VECTOR_JSON types - map to JSON
        return dt.JSON(nullable=True)

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
    elif typename == "TINY" and field_length == 1:
        # TINYINT(1) is commonly used as BOOLEAN in MySQL/SingleStoreDB
        # Note: SingleStoreDB BOOLEAN columns show field_length=4 at cursor level,
        # making them indistinguishable from TINYINT. The DESCRIBE-based schema
        # detection (via to_ibis method) can properly distinguish these types.
        typ = dt.Boolean
    elif typename == "VECTOR":
        # SingleStoreDB VECTOR type - typically used for AI/ML workloads
        # For now, map to Binary; could be enhanced to Array[Float32] in future
        typ = dt.Binary
    elif flags.is_set:
        # Sets are limited to strings in SingleStoreDB
        typ = dt.Array(dt.string)
    elif type_code in TEXT_TYPES:
        if flags.is_binary:
            typ = dt.Binary
        # For TEXT, MEDIUMTEXT, LONGTEXT (BLOB, MEDIUM_BLOB, LONG_BLOB)
        # don't include length as they are variable-length text types
        elif typename in ("BLOB", "MEDIUM_BLOB", "LONG_BLOB"):
            typ = dt.String  # No length parameter for unlimited text types
        else:
            # For VARCHAR, CHAR, etc. include the length
            typ = partial(dt.String, length=field_length // multi_byte_maximum_length)
    elif flags.is_timestamp or typename == "TIMESTAMP":
        # SingleStoreDB timestamps - note timezone handling
        # SingleStoreDB stores timestamps in UTC by default in columnstore tables
        typ = partial(dt.Timestamp, timezone="UTC", scale=scale or None)
    elif typename == "DATETIME":
        # DATETIME doesn't have timezone info in SingleStoreDB
        typ = partial(dt.Timestamp, scale=scale or None)
    elif typename == "JSON":
        # SingleStoreDB has enhanced JSON support with columnstore optimizations
        typ = dt.JSON
    elif typename == "GEOGRAPHY":
        # SingleStoreDB extended geospatial type
        typ = dt.Geometry
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
    # VECTOR type for machine learning and AI workloads
    "VECTOR": dt.Binary,  # Map to Binary for now, could be Array[Float32] in future
    # Extended types (SingleStoreDB-specific extensions)
    "GEOGRAPHY": dt.Geometry,  # Enhanced geospatial support
}


class SingleStoreDBType(SqlglotType):
    """SingleStoreDB data type implementation.

    SingleStoreDB uses the MySQL protocol but has additional features:
    - Enhanced JSON support with columnstore optimizations
    - VECTOR type for AI/ML workloads
    - GEOGRAPHY type for extended geospatial operations
    - ROWSTORE vs COLUMNSTORE table types with different optimizations

    Note on schema detection:
    SingleStoreDB has two schema detection paths with different capabilities:
    1. Cursor-based (_type_from_cursor_info): Uses raw cursor metadata but cannot
       distinguish BOOLEAN from TINYINT due to identical protocol-level representation
    2. DESCRIBE-based (to_ibis): Uses SQL DESCRIBE command and can properly distinguish
       types like BOOLEAN vs TINYINT based on type string parsing
    """

    dialect = "singlestore"  # SingleStoreDB uses SingleStore dialect in SQLGlot

    # SingleStoreDB-specific type mappings and defaults
    default_decimal_precision = 10
    default_decimal_scale = 0
    default_temporal_scale = None

    # Type mappings for SingleStoreDB-specific types
    _singlestore_type_mapping = {
        # Standard types (same as MySQL)
        **_type_mapping,
        # SingleStoreDB-specific enhancements
        "VECTOR": dt.Binary,  # Vector type for ML/AI (mapped to Binary for now)
        "GEOGRAPHY": dt.Geometry,  # Enhanced geospatial support
    }

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        """Convert SingleStoreDB type to Ibis type.

        Handles both standard MySQL types and SingleStoreDB-specific extensions.
        """
        if hasattr(typ, "this"):
            type_name = str(typ.this).upper()

            # Handle BOOLEAN type directly
            if type_name == "BOOLEAN":
                return dt.Boolean(nullable=nullable)

            # Handle TINYINT as Boolean - MySQL/SingleStoreDB convention
            if type_name.endswith("TINYINT"):
                # Check if it has explicit length parameter
                if hasattr(typ, "expressions") and typ.expressions:
                    # Extract length parameter from TINYINT(length)
                    length_param = typ.expressions[0]
                    if hasattr(length_param, "this") and hasattr(
                        length_param.this, "this"
                    ):
                        length = int(length_param.this.this)
                        if length == 1:
                            # TINYINT(1) is commonly used as BOOLEAN
                            return dt.Boolean(nullable=nullable)
                else:
                    # TINYINT without explicit length - in SingleStoreDB this often means BOOLEAN
                    # Check if it's likely a boolean context by falling back to the parent's handling
                    # but first try the _type_mapping which should handle TINY -> dt.Int8
                    pass  # Let it fall through to normal handling

            # Handle DATETIME with scale parameter specially
            # Note: type_name will be "TYPE.DATETIME", so check for endswith
            if (
                type_name.endswith("DATETIME")
                and hasattr(typ, "expressions")
                and typ.expressions
            ):
                # Extract scale from the first parameter
                scale_param = typ.expressions[0]
                if hasattr(scale_param, "this") and hasattr(scale_param.this, "this"):
                    scale = int(scale_param.this.this)
                    return dt.Timestamp(scale=scale or None, nullable=nullable)

            # Handle BIT types with length parameter
            if (
                type_name.endswith("BIT")
                and hasattr(typ, "expressions")
                and typ.expressions
            ):
                # Extract bit length from the first parameter
                length_param = typ.expressions[0]
                if hasattr(length_param, "this") and hasattr(length_param.this, "this"):
                    bit_length = int(length_param.this.this)
                    # Map bit length to appropriate integer type
                    if bit_length <= 8:
                        return dt.Int8(nullable=nullable)
                    elif bit_length <= 16:
                        return dt.Int16(nullable=nullable)
                    elif bit_length <= 32:
                        return dt.Int32(nullable=nullable)
                    elif bit_length <= 64:
                        return dt.Int64(nullable=nullable)
                    else:
                        raise ValueError(f"BIT({bit_length}) is not supported")

            # Handle DECIMAL types with precision and scale parameters
            if (
                type_name.endswith(("DECIMAL", "NEWDECIMAL"))
                and hasattr(typ, "expressions")
                and typ.expressions
            ):
                # Extract precision and scale from parameters
                if len(typ.expressions) >= 1:
                    precision_param = typ.expressions[0]
                    if hasattr(precision_param, "this") and hasattr(
                        precision_param.this, "this"
                    ):
                        precision = int(precision_param.this.this)

                        scale = 0  # Default scale
                        if len(typ.expressions) >= 2:
                            scale_param = typ.expressions[1]
                            if hasattr(scale_param, "this") and hasattr(
                                scale_param.this, "this"
                            ):
                                scale = int(scale_param.this.this)

                        return dt.Decimal(
                            precision=precision, scale=scale, nullable=nullable
                        )

            # Handle string types with length parameters (VARCHAR, CHAR)
            if (
                type_name.endswith(("VARCHAR", "CHAR"))
                and hasattr(typ, "expressions")
                and typ.expressions
            ):
                # Extract length from the first parameter
                length_param = typ.expressions[0]
                if hasattr(length_param, "this") and hasattr(length_param.this, "this"):
                    length = int(length_param.this.this)
                    return dt.String(length=length, nullable=nullable)

            # Handle binary types with length parameters (BINARY, VARBINARY)
            if (
                type_name.endswith(("BINARY", "VARBINARY"))
                and hasattr(typ, "expressions")
                and typ.expressions
            ):
                # Extract length from the first parameter
                length_param = typ.expressions[0]
                if hasattr(length_param, "this") and hasattr(length_param.this, "this"):
                    length = int(length_param.this.this)
                    return dt.Binary(
                        nullable=nullable
                    )  # Note: Ibis Binary doesn't store length

            # Extract just the type part (e.g., "DATETIME" from "TYPE.DATETIME")
            if "." in type_name:
                type_name = type_name.split(".")[-1]

            # Handle other SingleStoreDB-specific types
            if type_name in cls._singlestore_type_mapping:
                ibis_type = cls._singlestore_type_mapping[type_name]
                if callable(ibis_type):
                    return ibis_type(nullable=nullable)
                else:
                    return ibis_type(nullable=nullable)

        # Fall back to parent implementation for standard types
        return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        """Convert Ibis type to SingleStoreDB type.

        Handles conversion from Ibis types to SingleStoreDB SQL types,
        including support for SingleStoreDB-specific features.
        """
        # Handle SingleStoreDB-specific type conversions
        if isinstance(dtype, dt.JSON):
            # SingleStoreDB has enhanced JSON support
            return sge.DataType(this=sge.DataType.Type.JSON)
        elif isinstance(dtype, dt.Geometry):
            # Use GEOMETRY type (or GEOGRAPHY if available)
            return sge.DataType(this=sge.DataType.Type.GEOMETRY)
        elif isinstance(dtype, dt.Binary):
            # Could be BLOB or VECTOR type - default to BLOB
            return sge.DataType(this=sge.DataType.Type.BLOB)
        elif isinstance(dtype, dt.UUID):
            # SingleStoreDB doesn't support UUID natively, map to CHAR(36)
            return sge.DataType(
                this=sge.DataType.Type.CHAR, expressions=[sge.convert(36)]
            )
        elif isinstance(dtype, dt.Timestamp):
            # SingleStoreDB only supports DATETIME precision 0 or 6
            # Normalize precision to nearest supported value
            if dtype.scale is not None:
                if dtype.scale <= 3:
                    # Use precision 0 for scales 0-3
                    precision = 0
                else:
                    # Use precision 6 for scales 4-9
                    precision = 6

                if precision == 0:
                    return sge.DataType(this=sge.DataType.Type.DATETIME)
                else:
                    return sge.DataType(
                        this=sge.DataType.Type.DATETIME,
                        expressions=[sge.convert(precision)],
                    )
            else:
                # Default DATETIME without precision
                return sge.DataType(this=sge.DataType.Type.DATETIME)

        # Fall back to parent implementation for standard types
        return super().from_ibis(dtype)
