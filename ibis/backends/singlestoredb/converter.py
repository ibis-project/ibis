from __future__ import annotations

import datetime
import json

import ibis.expr.datatypes as dt
from ibis.formats.pandas import PandasData


class SingleStoreDBPandasData(PandasData):
    """Data converter for SingleStoreDB backend using pandas format."""

    @classmethod
    def convert_Time(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB TIME values to Python time objects."""
        import pandas as pd

        def convert(value):
            if value is None:
                return None

            # Handle Timedelta objects (from TIME operations)
            if isinstance(value, pd.Timedelta):
                total_seconds = int(value.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                microseconds = value.microseconds
                return datetime.time(
                    hour=hours % 24,  # Ensure we don't exceed 24 hours
                    minute=minutes,
                    second=seconds,
                    microsecond=microseconds,
                )

            # Handle timedelta64 objects
            elif hasattr(value, "components"):
                comps = value.components
                return datetime.time(
                    hour=comps.hours,
                    minute=comps.minutes,
                    second=comps.seconds,
                    microsecond=comps.milliseconds * 1000 + comps.microseconds,
                )

            # Handle datetime.time objects (already proper)
            elif isinstance(value, datetime.time):
                return value

            # Handle string representations
            elif isinstance(value, str):
                try:
                    # Parse HH:MM:SS or HH:MM:SS.ffffff format
                    if "." in value:
                        time_part, microsec_part = value.split(".")
                        microseconds = int(microsec_part.ljust(6, "0")[:6])
                    else:
                        time_part = value
                        microseconds = 0

                    parts = time_part.split(":")
                    if len(parts) >= 3:
                        return datetime.time(
                            hour=int(parts[0]) % 24,
                            minute=int(parts[1]),
                            second=int(parts[2]),
                            microsecond=microseconds,
                        )
                except (ValueError, IndexError):
                    pass

            return value

        return s.map(convert, na_action="ignore")

    @classmethod
    def convert_Timestamp(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB TIMESTAMP/DATETIME values."""
        if s.dtype == "object":
            # Handle SingleStoreDB zero timestamps
            s = s.replace("0000-00-00 00:00:00", None)
        return super().convert_Timestamp(s, dtype, pandas_type)

    @classmethod
    def convert_Date(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB DATE values."""
        import pandas as pd

        def convert_date(value):
            if value is None:
                return None

            # Handle bytes objects (from STR_TO_DATE)
            if isinstance(value, bytes):
                try:
                    date_str = value.decode("utf-8")
                    return pd.to_datetime(date_str).date()
                except (UnicodeDecodeError, ValueError):
                    return None

            # Handle string representations
            elif isinstance(value, str):
                if value == "0000-00-00":
                    return None
                try:
                    return pd.to_datetime(value).date()
                except ValueError:
                    return None

            # Handle datetime objects
            elif hasattr(value, "date"):
                return value.date()

            return value

        if s.dtype == "object":
            # Handle SingleStoreDB zero dates and bytes
            s = s.map(convert_date, na_action="ignore")
            return s

        return super().convert_Date(s, dtype, pandas_type)

    @classmethod
    def convert_JSON(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB JSON values.

        SingleStoreDB has enhanced JSON support with columnstore optimizations.
        JSON values can be stored efficiently and queried with optimized functions.

        For PyArrow compatibility, we return JSON as strings rather than parsed objects.
        """

        def convert_json(value):
            if value is None:
                # Convert SQL NULL to JSON null string for proper handling
                return "null"

            # Try to determine if value is already a valid JSON string
            if isinstance(value, str):
                try:
                    # Test if it's already valid JSON
                    json.loads(value)
                    return value
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, so JSON-encode it as a string
                    return json.dumps(value)
            else:
                # For all other types (dict, list, bool, int, float), JSON-encode them
                return json.dumps(value)

        return s.map(convert_json)

    @classmethod
    def convert_Binary(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB binary data including VECTOR type."""

        def convert_binary(value):
            if value is None:
                return None
            # Handle VECTOR type data if it comes as bytes
            if isinstance(value, bytes):
                return value
            # Handle string representation
            elif isinstance(value, str):
                try:
                    return bytes.fromhex(value)
                except ValueError:
                    return value.encode("utf-8")
            return value

        return s.map(convert_binary, na_action="ignore")

    @classmethod
    def convert_Decimal(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB DECIMAL/NUMERIC values with proper NULL handling."""
        # Handle SingleStoreDB NULL decimals
        if s.dtype == "object":
            s = s.replace("", None)  # Empty strings as NULL
        return super().convert_Decimal(s, dtype, pandas_type)

    @classmethod
    def convert_String(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB string types with proper NULL handling."""
        # SingleStoreDB may return empty strings for some NULL cases
        if hasattr(dtype, "nullable") and dtype.nullable:
            s = s.replace("", None)
        return super().convert_String(s, dtype, pandas_type)

    @classmethod
    def convert_Boolean(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB TINYINT(1) boolean values to proper booleans.

        SingleStoreDB uses TINYINT(1) for boolean columns, which return integer
        values (0, 1) that need to be converted to proper boolean types for PyArrow.
        """

        def convert_bool(value):
            if value is None:
                return None
            # Convert integer values (0, 1) to boolean
            if isinstance(value, int):
                return bool(value)
            # Handle string representations
            elif isinstance(value, str):
                if value.lower() in ("true", "1", "yes", "on"):
                    return True
                elif value.lower() in ("false", "0", "no", "off"):
                    return False
                else:
                    return None
            # Already boolean
            elif isinstance(value, bool):
                return value
            else:
                return bool(value) if value is not None else None

        return s.map(convert_bool, na_action="ignore")

    @classmethod
    def handle_null_value(cls, value, target_type):
        """Handle NULL values consistently across all SingleStoreDB types.

        SingleStoreDB may represent NULLs differently depending on the type
        and storage format (ROWSTORE vs COLUMNSTORE).
        """
        if value is None:
            return None

        # Handle different NULL representations
        if isinstance(value, str):
            # Common NULL string representations
            if value in ("", "NULL", "null", "0000-00-00", "0000-00-00 00:00:00"):
                return None

        # Handle numeric zero values that might represent NULL for date/timestamp types
        if target_type.is_date() or target_type.is_timestamp():
            if value == 0:
                return None

        return value

    @classmethod
    def _get_type_name(cls, type_code: int) -> str:
        """Get type name from SingleStoreDB type code.

        SingleStoreDB uses MySQL protocol, so type codes are the same.
        """
        type_map = {
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
            14: "NEWDATE",
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
        return type_map.get(type_code, "UNKNOWN")

    @classmethod
    def convert_SingleStoreDB_type(cls, typename: str) -> dt.DataType:
        """Convert a SingleStoreDB type name to an Ibis data type.

        Handles both standard MySQL-compatible types and SingleStoreDB-specific extensions.
        """
        typename = typename.upper()

        # Numeric types
        if typename in ("TINY", "TINYINT"):
            return dt.int8
        elif typename in ("SHORT", "SMALLINT"):
            return dt.int16
        elif typename in ("LONG", "INT", "INTEGER"):
            return dt.int32
        elif typename in ("LONGLONG", "BIGINT"):
            return dt.int64
        elif typename == "FLOAT":
            return dt.float32
        elif typename == "DOUBLE":
            return dt.float64
        elif typename in ("DECIMAL", "NEWDECIMAL"):
            return dt.decimal
        elif typename == "BIT":
            return dt.int8  # For BIT(1), larger BIT fields map to larger ints
        elif typename == "YEAR":
            return dt.uint8

        # String types
        elif typename in ("VARCHAR", "VAR_STRING", "CHAR"):
            return dt.string
        elif typename in ("STRING", "TEXT"):
            return dt.string
        elif typename == "ENUM":
            return dt.string

        # Temporal types
        elif typename == "DATE":
            return dt.date
        elif typename == "TIME":
            return dt.time
        elif typename in ("DATETIME", "TIMESTAMP"):
            return dt.timestamp

        # Binary types
        elif typename in ("BLOB", "TINY_BLOB", "MEDIUM_BLOB", "LONG_BLOB"):
            return dt.binary
        elif typename in ("BINARY", "VARBINARY"):
            return dt.binary

        # Special types
        elif typename == "JSON":
            # SingleStoreDB has enhanced JSON support with columnstore optimizations
            return dt.json
        elif typename == "GEOMETRY":
            return dt.geometry  # Use geometry type instead of binary
        elif typename == "NULL":
            return dt.null

        # Collection types
        elif typename == "SET":
            return dt.Array(dt.string)  # SET is like an array of strings

        # SingleStoreDB-specific types
        elif typename == "VECTOR":
            # Vector type for ML/AI workloads - map to binary for now
            # In future could be Array[Float32] with proper vector support
            return dt.binary
        elif typename == "GEOGRAPHY":
            # Enhanced geospatial support
            return dt.geometry

        else:
            # Default to string for unknown types
            return dt.string
