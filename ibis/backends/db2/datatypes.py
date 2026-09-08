"""Db2 data type mappings for Ibis."""

from __future__ import annotations

import ibis.expr.datatypes as dt

# Mapping from Db2 type names to Ibis types
DB2_TO_IBIS_TYPE = {
    # Integer types
    "SMALLINT": dt.int16,
    "INTEGER": dt.int32,
    "INT": dt.int32,
    "BIGINT": dt.int64,
    # Floating point types
    "REAL": dt.float32,
    "FLOAT": dt.float64,
    "DOUBLE": dt.float64,
    "DOUBLE PRECISION": dt.float64,
    "DECFLOAT": dt.float64,
    # Decimal types
    "DECIMAL": dt.Decimal,
    "NUMERIC": dt.Decimal,
    "DEC": dt.Decimal,
    # String types
    "CHARACTER": dt.string,
    "CHAR": dt.string,
    "VARCHAR": dt.string,
    "CHARACTER VARYING": dt.string,
    "LONG VARCHAR": dt.string,
    "CLOB": dt.string,
    "CHAR () FOR BIT DATA": dt.binary,
    "VARCHAR () FOR BIT DATA": dt.binary,
    "LONG VARCHAR FOR BIT DATA": dt.binary,
    # Binary types
    "BINARY": dt.binary,
    "VARBINARY": dt.binary,
    "BLOB": dt.binary,
    # Date/Time types
    "DATE": dt.date,
    "TIME": dt.time,
    "TIMESTAMP": dt.timestamp,
    # Boolean type
    "BOOLEAN": dt.boolean,
    # XML type
    "XML": dt.string,
    # Graphic types (for DBCS)
    "GRAPHIC": dt.string,
    "VARGRAPHIC": dt.string,
    "LONG VARGRAPHIC": dt.string,
    "DBCLOB": dt.string,
}


def parse_db2_type(type_string: str) -> dt.DataType:
    """Parse a Db2 type string into an Ibis data type.

    Parameters
    ----------
    type_string : str
        Db2 type string (e.g., "VARCHAR(100)", "DECIMAL(10,2)")

    Returns
    -------
    dt.DataType
        Corresponding Ibis data type

    Examples
    --------
    >>> parse_db2_type("VARCHAR(100)")
    String(length=None, nullable=True)
    >>> parse_db2_type("DECIMAL(10,2)")
    Decimal(precision=10, scale=2, nullable=True)
    """
    type_string = type_string.upper().strip()

    # Handle parameterized types
    if "(" in type_string:
        base_type: str = type_string.split("(")[0].strip()
        params = type_string.split("(")[1].rstrip(")").split(",")
        params = [p.strip() for p in params]

        if base_type in ("DECIMAL", "NUMERIC", "DEC"):
            precision = int(params[0]) if params else 10
            scale = int(params[1]) if len(params) > 1 else 0
            return dt.Decimal(precision=precision, scale=scale, nullable=True)
        elif base_type in ("VARCHAR", "CHARACTER VARYING", "CHAR", "CHARACTER"):
            # Return string type regardless of length
            return dt.string
        elif base_type in ("VARBINARY", "BINARY"):
            return dt.binary
        elif base_type == "TIMESTAMP":
            # Db2 timestamps can have precision
            return dt.timestamp
        else:
            # For other parameterized types, use base type
            return DB2_TO_IBIS_TYPE.get(base_type, dt.string)

    # Handle non-parameterized types
    return DB2_TO_IBIS_TYPE.get(type_string, dt.string)


def ibis_type_to_db2_type(ibis_type: dt.DataType) -> str:
    """Convert an Ibis data type to a Db2 type string.

    Parameters
    ----------
    ibis_type : dt.DataType
        Ibis data type

    Returns
    -------
    str
        Db2 type string

    Examples
    --------
    >>> ibis_type_to_db2_type(dt.int32)
    'INTEGER'
    >>> ibis_type_to_db2_type(dt.Decimal(10, 2))
    'DECIMAL(10, 2)'
    """
    if isinstance(ibis_type, dt.Int8):
        return "SMALLINT"
    elif isinstance(ibis_type, dt.Int16):
        return "SMALLINT"
    elif isinstance(ibis_type, dt.Int32):
        return "INTEGER"
    elif isinstance(ibis_type, dt.Int64):
        return "BIGINT"
    elif isinstance(ibis_type, dt.UInt8):
        return "SMALLINT"
    elif isinstance(ibis_type, dt.UInt16):
        return "INTEGER"
    elif isinstance(ibis_type, dt.UInt32):
        return "BIGINT"
    elif isinstance(ibis_type, dt.UInt64):
        return "BIGINT"
    elif isinstance(ibis_type, dt.Float32):
        return "REAL"
    elif isinstance(ibis_type, dt.Float64):
        return "DOUBLE"
    elif isinstance(ibis_type, dt.Decimal):
        precision = ibis_type.precision or 10
        scale = ibis_type.scale or 0
        return f"DECIMAL({precision}, {scale})"
    elif isinstance(ibis_type, dt.String):
        return "VARCHAR(32672)"  # Max VARCHAR length in Db2
    elif isinstance(ibis_type, dt.Binary):
        return "VARBINARY(32672)"
    elif isinstance(ibis_type, dt.Date):
        return "DATE"
    elif isinstance(ibis_type, dt.Time):
        return "TIME"
    elif isinstance(ibis_type, dt.Timestamp):
        return "TIMESTAMP"
    elif isinstance(ibis_type, dt.Boolean):
        return "BOOLEAN"
    elif isinstance(ibis_type, dt.JSON):
        return "CLOB"  # Store JSON as CLOB
    elif isinstance(ibis_type, dt.UUID):
        return "CHAR(36)"
    elif isinstance(ibis_type, dt.Array):
        # Db2 doesn't have a native array type — use CLOB for JSON representation
        return "CLOB"
    elif isinstance(ibis_type, dt.Map):
        # Db2 doesn't have a native map type — use CLOB for JSON representation
        return "CLOB"
    elif isinstance(ibis_type, dt.Struct):
        # Db2 doesn't have a native struct type — use CLOB for JSON representation
        return "CLOB"
    else:
        # Default to VARCHAR for unknown types
        return "VARCHAR(32672)"


# Type code mappings for ibm_db_dbi
DB2_TYPE_CODES = {
    384: dt.date,  # DATE
    388: dt.time,  # TIME
    392: dt.timestamp,  # TIMESTAMP
    448: dt.string,  # VARCHAR
    452: dt.string,  # CHAR
    456: dt.string,  # LONG VARCHAR
    460: dt.string,  # CLOB
    464: dt.binary,  # BLOB
    468: dt.binary,  # BINARY
    472: dt.binary,  # VARBINARY
    480: dt.float64,  # FLOAT
    484: dt.float64,  # DOUBLE
    492: dt.Decimal,  # DECIMAL
    496: dt.int32,  # INTEGER
    500: dt.int16,  # SMALLINT
    504: dt.int64,  # BIGINT
    908: dt.boolean,  # BOOLEAN
}


def type_code_to_ibis_type(
    type_code: int, precision: int = 0, scale: int = 0
) -> dt.DataType:
    """Convert a Db2 type code to an Ibis data type.

    Parameters
    ----------
    type_code : int
        Db2 type code from cursor description
    precision : int, default 0
        Precision for decimal types
    scale : int, default 0
        Scale for decimal types

    Returns
    -------
    dt.DataType
        Corresponding Ibis data type
    """
    base_type = DB2_TYPE_CODES.get(type_code, dt.string)

    if type_code == 492 and precision > 0:  # DECIMAL
        return dt.Decimal(precision=precision, scale=scale, nullable=True)

    return base_type
