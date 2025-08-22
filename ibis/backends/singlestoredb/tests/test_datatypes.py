"""Tests for SingleStoreDB data type mappings and conversions."""

from __future__ import annotations

import datetime
from functools import partial

import pytest

import ibis.expr.datatypes as dt
from ibis.backends.singlestoredb.converter import SingleStoreDBPandasData
from ibis.backends.singlestoredb.datatypes import (
    SingleStoreDBType,
    _type_from_cursor_info,
    _type_mapping,
)


class TestSingleStoreDBDataTypes:
    """Test SingleStoreDB data type mappings."""

    def test_basic_type_mappings(self):
        """Test that basic SingleStoreDB types map to correct Ibis types."""
        expected_mappings = {
            # Numeric types
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
            "VECTOR": dt.Binary,
            "GEOGRAPHY": dt.Geometry,
        }

        for singlestore_type, expected_ibis_type in expected_mappings.items():
            actual_type = _type_mapping[singlestore_type]

            # Handle partial comparison for SET type
            if isinstance(expected_ibis_type, partial) and isinstance(
                actual_type, partial
            ):
                assert actual_type.func == expected_ibis_type.func
                assert actual_type.args == expected_ibis_type.args
            else:
                assert actual_type == expected_ibis_type

    def test_singlestoredb_specific_types(self):
        """Test SingleStoreDB-specific type extensions."""
        # Test VECTOR type
        assert "VECTOR" in _type_mapping
        assert _type_mapping["VECTOR"] == dt.Binary

        # Test GEOGRAPHY type
        assert "GEOGRAPHY" in _type_mapping
        assert _type_mapping["GEOGRAPHY"] == dt.Geometry

    def test_decimal_type_with_precision_and_scale(self):
        """Test DECIMAL type with precision and scale parameters."""
        # Mock cursor info for DECIMAL type
        result = _type_from_cursor_info(
            flags=0,
            type_code=0,  # DECIMAL type code
            field_length=10,
            scale=2,
            multi_byte_maximum_length=1,
        )

        assert isinstance(result, dt.Decimal)
        assert result.precision == 8  # Calculated precision
        assert result.scale == 2
        assert result.nullable is True

    def test_bit_type_field_length_mapping(self):
        """Test BIT type maps to appropriate integer type based on field length."""
        test_cases = [
            (1, dt.Int8),
            (8, dt.Int8),
            (9, dt.Int16),
            (16, dt.Int16),
            (17, dt.Int32),
            (32, dt.Int32),
            (33, dt.Int64),
            (64, dt.Int64),
        ]

        for field_length, expected_type in test_cases:
            result = _type_from_cursor_info(
                flags=0,
                type_code=16,  # BIT type code
                field_length=field_length,
                scale=0,
                multi_byte_maximum_length=1,
            )
            assert isinstance(result, expected_type)

    def test_vector_type_handling(self):
        """Test VECTOR type handling from cursor info."""
        result = _type_from_cursor_info(
            flags=0,
            type_code=256,  # Hypothetical VECTOR type code
            field_length=1024,  # Vector dimension
            scale=0,
            multi_byte_maximum_length=1,
        )

        assert isinstance(result, dt.Binary)
        assert result.nullable is True

    def test_timestamp_with_timezone(self):
        """Test TIMESTAMP type includes UTC timezone by default."""
        result = _type_from_cursor_info(
            flags=1024,  # TIMESTAMP flag
            type_code=7,  # TIMESTAMP type code
            field_length=0,
            scale=6,  # microsecond precision
            multi_byte_maximum_length=1,
        )

        assert isinstance(result, dt.Timestamp)
        assert result.timezone == "UTC"
        assert result.scale == 6
        assert result.nullable is True

    def test_datetime_without_timezone(self):
        """Test DATETIME type has no timezone."""
        result = _type_from_cursor_info(
            flags=0,
            type_code=12,  # DATETIME type code
            field_length=0,
            scale=3,
            multi_byte_maximum_length=1,
        )

        assert isinstance(result, dt.Timestamp)
        assert result.timezone is None
        assert result.scale == 3

    def test_json_type_handling(self):
        """Test JSON type is properly mapped."""
        result = _type_from_cursor_info(
            flags=0,
            type_code=245,  # JSON type code
            field_length=0,
            scale=0,
            multi_byte_maximum_length=1,
        )

        assert isinstance(result, dt.JSON)
        assert result.nullable is True

    def test_set_type_as_array(self):
        """Test SET type is mapped to Array[String]."""
        result = _type_from_cursor_info(
            flags=2048,  # SET flag
            type_code=248,  # SET type code
            field_length=0,
            scale=0,
            multi_byte_maximum_length=1,
        )

        assert isinstance(result, dt.Array)
        assert isinstance(result.value_type, dt.String)

    def test_unsigned_integer_mapping(self):
        """Test unsigned integer types are properly mapped."""
        result = _type_from_cursor_info(
            flags=32,  # UNSIGNED flag
            type_code=3,  # LONG type code (INT32)
            field_length=0,
            scale=0,
            multi_byte_maximum_length=1,
        )

        assert isinstance(result, dt.UInt32)

    def test_binary_vs_string_text_types(self):
        """Test binary flag determines if text types become Binary or String."""
        # Binary text type
        binary_result = _type_from_cursor_info(
            flags=128,  # BINARY flag
            type_code=252,  # BLOB type code
            field_length=255,
            scale=0,
            multi_byte_maximum_length=1,
        )
        assert isinstance(binary_result, dt.Binary)

        # String text type
        string_result = _type_from_cursor_info(
            flags=0,  # No BINARY flag
            type_code=254,  # STRING type code
            field_length=255,
            scale=0,
            multi_byte_maximum_length=1,
        )
        assert isinstance(string_result, dt.String)
        assert string_result.length == 255


class TestSingleStoreDBTypeClass:
    """Test the SingleStoreDBType class."""

    def test_singlestore_type_mapping_includes_all_types(self):
        """Test that SingleStoreDBType includes all expected mappings."""
        type_mapper = SingleStoreDBType()

        # Should include all standard mappings plus SingleStoreDB-specific ones
        expected_keys = set(_type_mapping.keys()) | {"VECTOR", "GEOGRAPHY"}
        actual_keys = set(type_mapper._singlestore_type_mapping.keys())

        assert expected_keys.issubset(actual_keys)

    def test_from_ibis_json_type(self):
        """Test conversion from Ibis JSON type to SingleStoreDB."""
        json_dtype = dt.JSON()
        result = SingleStoreDBType.from_ibis(json_dtype)
        # Should generate appropriate SQL representation
        assert result is not None

    def test_from_ibis_geometry_type(self):
        """Test conversion from Ibis Geometry type to SingleStoreDB."""
        geometry_dtype = dt.Geometry()
        result = SingleStoreDBType.from_ibis(geometry_dtype)
        assert result is not None

    def test_from_ibis_binary_type(self):
        """Test conversion from Ibis Binary type to SingleStoreDB."""
        binary_dtype = dt.Binary()
        result = SingleStoreDBType.from_ibis(binary_dtype)
        assert result is not None


class TestSingleStoreDBConverter:
    """Test the SingleStoreDB pandas data converter."""

    def test_convert_time_values(self):
        """Test TIME value conversion with timedelta components."""
        import pandas as pd

        # Create a sample timedelta
        timedelta_val = pd.Timedelta(
            hours=10, minutes=30, seconds=45, milliseconds=123, microseconds=456
        )
        series = pd.Series([timedelta_val, None])

        result = SingleStoreDBPandasData.convert_Time(series, dt.time, None)

        expected_time = datetime.time(hour=10, minute=30, second=45, microsecond=123456)
        assert result.iloc[0] == expected_time
        assert pd.isna(result.iloc[1])

    def test_convert_timestamp_zero_handling(self):
        """Test TIMESTAMP conversion handles zero timestamps."""
        import pandas as pd

        series = pd.Series(["2023-01-01 10:30:45", "0000-00-00 00:00:00", None])

        result = SingleStoreDBPandasData.convert_Timestamp(series, dt.timestamp, None)

        assert not pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])  # Zero timestamp should become None
        assert pd.isna(result.iloc[2])

    def test_convert_date_zero_handling(self):
        """Test DATE conversion handles zero dates."""
        import pandas as pd

        series = pd.Series(["2023-01-01", "0000-00-00", None])

        result = SingleStoreDBPandasData.convert_Date(series, dt.date, None)

        assert not pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])  # Zero date should become None
        assert pd.isna(result.iloc[2])

    def test_convert_json_values(self):
        """Test JSON value conversion."""
        import pandas as pd

        json_data = ['{"key": "value"}', '{"number": 42}', "invalid json", None]
        series = pd.Series(json_data)

        result = SingleStoreDBPandasData.convert_JSON(series, dt.json, None)

        assert result.iloc[0] == {"key": "value"}
        assert result.iloc[1] == {"number": 42}
        assert result.iloc[2] == "invalid json"  # Invalid JSON returns as string
        assert pd.isna(result.iloc[3])

    def test_convert_binary_values(self):
        """Test binary value conversion including VECTOR type support."""
        import pandas as pd

        binary_data = [
            b"binary_data",
            "48656c6c6f",
            "Hello",
            None,
        ]  # bytes, hex, string, None
        series = pd.Series(binary_data)

        result = SingleStoreDBPandasData.convert_Binary(series, dt.binary, None)

        assert result.iloc[0] == b"binary_data"
        assert result.iloc[1] == bytes.fromhex("48656c6c6f")
        assert result.iloc[2] == b"Hello"
        assert pd.isna(result.iloc[3])

    def test_convert_decimal_null_handling(self):
        """Test DECIMAL conversion handles NULL values."""
        import pandas as pd

        series = pd.Series(["123.45", "", "67.89", None], dtype=object)

        result = SingleStoreDBPandasData.convert_Decimal(series, dt.decimal, None)

        # Empty string should be converted to None for nullable decimals
        assert not pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])  # Empty string as NULL
        assert not pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])

    def test_handle_null_value_method(self):
        """Test the general null value handler."""
        converter = SingleStoreDBPandasData()

        # Test various NULL representations
        assert converter.handle_null_value(None, dt.string) is None
        assert converter.handle_null_value("", dt.string) is None
        assert converter.handle_null_value("NULL", dt.string) is None
        assert converter.handle_null_value("null", dt.string) is None
        assert converter.handle_null_value("0000-00-00", dt.date) is None
        assert converter.handle_null_value("0000-00-00 00:00:00", dt.timestamp) is None
        assert converter.handle_null_value(0, dt.date) is None

        # Test non-NULL values
        assert converter.handle_null_value("valid_string", dt.string) == "valid_string"
        assert converter.handle_null_value(123, dt.int32) == 123

    def test_get_type_name_mapping(self):
        """Test type code to name mapping."""
        converter = SingleStoreDBPandasData()

        # Test standard MySQL-compatible types
        assert converter._get_type_name(0) == "DECIMAL"
        assert converter._get_type_name(1) == "TINY"
        assert converter._get_type_name(245) == "JSON"
        assert converter._get_type_name(255) == "GEOMETRY"

        # Test unknown type code
        assert converter._get_type_name(999) == "UNKNOWN"

    def test_convert_singlestoredb_type_method(self):
        """Test the SingleStoreDB type name to Ibis type conversion."""
        converter = SingleStoreDBPandasData()

        # Test standard types
        assert converter.convert_SingleStoreDB_type("INT") == dt.int32
        assert converter.convert_SingleStoreDB_type("VARCHAR") == dt.string
        assert converter.convert_SingleStoreDB_type("JSON") == dt.json
        assert converter.convert_SingleStoreDB_type("GEOMETRY") == dt.geometry

        # Test SingleStoreDB-specific types
        assert converter.convert_SingleStoreDB_type("VECTOR") == dt.binary
        assert converter.convert_SingleStoreDB_type("GEOGRAPHY") == dt.geometry

        # Test case insensitivity
        assert converter.convert_SingleStoreDB_type("varchar") == dt.string
        assert converter.convert_SingleStoreDB_type("Vector") == dt.binary

        # Test unknown type defaults to string
        assert converter.convert_SingleStoreDB_type("UNKNOWN_TYPE") == dt.string


if __name__ == "__main__":
    pytest.main([__file__])
