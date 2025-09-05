from __future__ import annotations

import datetime
import json

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
        import pandas as pd

        def convert_timestamp(value):
            if value is None:
                return None

            # Handle bytes objects (from STR_TO_DATE operations)
            if isinstance(value, bytes):
                try:
                    timestamp_str = value.decode("utf-8")
                    return pd.to_datetime(timestamp_str)
                except (UnicodeDecodeError, ValueError):
                    return None

            # Handle zero timestamps
            if isinstance(value, str) and value == "0000-00-00 00:00:00":
                return None

            return value

        if s.dtype == "object":
            # Handle SingleStoreDB zero timestamps and bytes
            s = s.map(convert_timestamp, na_action="ignore")

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

        For compatibility with tests and direct usage, we return parsed JSON objects.
        """

        def convert_json(value):
            if value is None:
                return None

            # Try to parse JSON string into Python object
            if isinstance(value, str):
                try:
                    # Parse valid JSON into Python object
                    return json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, return as string
                    return value
            else:
                # For non-string types (dict, list, etc.), return as-is
                return value

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
        # NOTE: Do not convert empty strings to None for JSON operations
        # Empty strings are valid JSON string values and should be preserved
        # Only convert empty strings to None in specific contexts where SingleStoreDB
        # returns empty strings to represent NULL values (e.g., some legacy column types)
        # For now, we preserve empty strings to fix JSON unwrap operations
        return super().convert_String(s, dtype, pandas_type)

    @classmethod
    def convert_Array(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB SET values to arrays.

        SET columns in SingleStoreDB return comma-separated string values
        that need to be split into arrays.
        """

        def convert_set(value):
            if value is None:
                return None

            # Handle string values (typical for SET columns)
            if isinstance(value, str):
                if not value:  # Empty string
                    return []
                # Split on comma and strip whitespace
                return [item.strip() for item in value.split(",") if item.strip()]

            # If already a list/array, return as-is
            if isinstance(value, (list, tuple)):
                return list(value)

            return value

        return s.map(convert_set, na_action="ignore")

    def handle_null_value(self, value, dtype):
        """Handle various NULL representations."""
        import ibis.expr.datatypes as dt

        # Direct None values
        if value is None:
            return None

        # Empty string as NULL for string types
        if isinstance(value, str) and value == "":
            return None

        # "NULL" and "null" strings as NULL
        if isinstance(value, str) and value.upper() == "NULL":
            return None

        # Zero timestamps/dates as NULL for temporal types
        if isinstance(dtype, (dt.Date, dt.Timestamp)):
            if value in {"0000-00-00", "0000-00-00 00:00:00"}:
                return None
            if isinstance(value, (int, float)) and value == 0:
                return None

        # Return the value as-is if not NULL
        return value

    def _get_type_name(self, type_code):
        """Map SingleStoreDB type codes to type names."""
        # SingleStoreDB type code mappings
        type_code_map = {
            0: "DECIMAL",
            1: "TINY",
            2: "SHORT",
            3: "LONG",
            4: "FLOAT",
            5: "DOUBLE",
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
            # SingleStoreDB-specific types
            1001: "BSON",
            # Vector JSON types
            2001: "FLOAT32_VECTOR_JSON",
            2002: "FLOAT64_VECTOR_JSON",
            2003: "INT8_VECTOR_JSON",
            2004: "INT16_VECTOR_JSON",
            2005: "INT32_VECTOR_JSON",
            2006: "INT64_VECTOR_JSON",
            # Vector binary types
            3001: "FLOAT32_VECTOR",
            3002: "FLOAT64_VECTOR",
            3003: "INT8_VECTOR",
            3004: "INT16_VECTOR",
            3005: "INT32_VECTOR",
            3006: "INT64_VECTOR",
        }

        return type_code_map.get(type_code, "UNKNOWN")

    def convert_SingleStoreDB_type(self, type_name):
        """Convert SingleStoreDB type names to Ibis data types."""
        import ibis.expr.datatypes as dt
        from ibis.backends.singlestoredb.datatypes import _type_mapping

        # Normalize type name to uppercase
        normalized_name = type_name.upper()

        # Use the existing type mapping first
        ibis_type = _type_mapping.get(normalized_name)
        if ibis_type is not None:
            # Handle partials (like SET type)
            if hasattr(ibis_type, "func"):
                return ibis_type()  # Call the partial function
            # Return instance for classes
            if isinstance(ibis_type, type):
                return ibis_type()
            return ibis_type

        # Common SQL type name aliases
        sql_aliases = {
            "INT": dt.int32,
            "INTEGER": dt.int32,
            "BIGINT": dt.int64,
            "SMALLINT": dt.int16,
            "TINYINT": dt.int8,
            "VARCHAR": dt.string,
            "CHAR": dt.string,
            "TEXT": dt.string,
            "MEDIUMTEXT": dt.string,
            "LONGTEXT": dt.string,
            "BINARY": dt.binary,
            "VARBINARY": dt.binary,
            "TIMESTAMP": dt.timestamp,
            "DATETIME": dt.timestamp,
            "DATE": dt.date,
            "TIME": dt.time,
            "DECIMAL": dt.decimal,
            "NUMERIC": dt.decimal,
            "FLOAT": dt.float32,
            "DOUBLE": dt.float64,
            "REAL": dt.float64,
        }

        ibis_type = sql_aliases.get(normalized_name)
        if ibis_type is not None:
            return ibis_type

        # SingleStoreDB-specific mappings
        singlestore_specific = {
            "VECTOR": dt.binary,
            "BSON": dt.JSON,
            "GEOGRAPHY": dt.geometry,
            # Vector binary types
            "FLOAT32_VECTOR": dt.binary,
            "FLOAT64_VECTOR": dt.binary,
            "INT8_VECTOR": dt.binary,
            "INT16_VECTOR": dt.binary,
            "INT32_VECTOR": dt.binary,
            "INT64_VECTOR": dt.binary,
            # Vector JSON types
            "FLOAT32_VECTOR_JSON": dt.JSON,
            "FLOAT64_VECTOR_JSON": dt.JSON,
            "INT8_VECTOR_JSON": dt.JSON,
            "INT16_VECTOR_JSON": dt.JSON,
            "INT32_VECTOR_JSON": dt.JSON,
            "INT64_VECTOR_JSON": dt.JSON,
        }

        ibis_type = singlestore_specific.get(normalized_name)
        if ibis_type is not None:
            return ibis_type

        # Default to string for unknown types
        return dt.string
