from __future__ import annotations

import datetime

import ibis.expr.datatypes as dt
from ibis.formats.pandas import PandasData


class SingleStoreDBPandasData(PandasData):
    """Data converter for SingleStoreDB backend using pandas format."""

    @classmethod
    def convert_Time(cls, s, dtype, pandas_type):
        """Convert SingleStoreDB TIME values to Python time objects."""

        def convert(timedelta):
            if timedelta is None:
                return None
            comps = timedelta.components
            return datetime.time(
                hour=comps.hours,
                minute=comps.minutes,
                second=comps.seconds,
                microsecond=comps.milliseconds * 1000 + comps.microseconds,
            )

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
        if s.dtype == "object":
            # Handle SingleStoreDB zero dates
            s = s.replace("0000-00-00", None)
        return super().convert_Date(s, dtype, pandas_type)

    @classmethod
    def _get_type_name(cls, type_code: int) -> str:
        """Get type name from MySQL/SingleStoreDB type code.

        SingleStoreDB uses MySQL protocol, so type codes are the same.
        """
        # MySQL field type constants
        # These are the same for SingleStoreDB due to protocol compatibility
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
        """Convert a SingleStoreDB type name to an Ibis data type."""
        typename = typename.upper()

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
        elif typename in ("VARCHAR", "VAR_STRING"):
            return dt.string
        elif typename == "STRING":
            return dt.string
        elif typename == "DATE":
            return dt.date
        elif typename == "TIME":
            return dt.time
        elif typename in ("DATETIME", "TIMESTAMP"):
            return dt.timestamp
        elif typename == "YEAR":
            return dt.uint8
        elif typename in ("BLOB", "TINY_BLOB", "MEDIUM_BLOB", "LONG_BLOB"):
            return dt.binary
        elif typename == "BIT":
            return dt.int8  # For BIT(1), larger BIT fields map to larger ints
        elif typename == "JSON":
            return dt.json
        elif typename == "ENUM":
            return dt.string
        elif typename == "SET":
            return dt.Array(dt.string)  # SET is like an array of strings
        elif typename == "GEOMETRY":
            return dt.binary  # Treat geometry as binary for now
        elif typename == "NULL":
            return dt.null
        else:
            # Default to string for unknown types
            return dt.string
