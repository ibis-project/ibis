"""Tests for DB2 data type mappings."""

from __future__ import annotations

import ibis.expr.datatypes as dt

from ibis.backends.db2.datatypes import (
    ibis_type_to_db2_type,
    parse_db2_type,
    type_code_to_ibis_type,
)


class TestParseDB2Type:
    """Tests for parse_db2_type function."""

    def test_integer_types(self):
        """Test parsing of integer types."""
        assert parse_db2_type("SMALLINT") == dt.int16
        assert parse_db2_type("INTEGER") == dt.int32
        assert parse_db2_type("INT") == dt.int32
        assert parse_db2_type("BIGINT") == dt.int64

    def test_float_types(self):
        """Test parsing of floating point types."""
        assert parse_db2_type("REAL") == dt.float32
        assert parse_db2_type("FLOAT") == dt.float64
        assert parse_db2_type("DOUBLE") == dt.float64
        assert parse_db2_type("DOUBLE PRECISION") == dt.float64

    def test_decimal_types(self):
        """Test parsing of decimal types."""
        result = parse_db2_type("DECIMAL(10,2)")
        assert isinstance(result, dt.Decimal)
        assert result.precision == 10
        assert result.scale == 2

        result = parse_db2_type("NUMERIC(15,3)")
        assert isinstance(result, dt.Decimal)
        assert result.precision == 15
        assert result.scale == 3

    def test_string_types(self):
        """Test parsing of string types."""
        assert parse_db2_type("VARCHAR(100)") == dt.string
        assert parse_db2_type("CHAR(10)") == dt.string
        assert parse_db2_type("CLOB") == dt.string
        assert parse_db2_type("CHARACTER VARYING(50)") == dt.string

    def test_binary_types(self):
        """Test parsing of binary types."""
        assert parse_db2_type("BINARY(10)") == dt.binary
        assert parse_db2_type("VARBINARY(100)") == dt.binary
        assert parse_db2_type("BLOB") == dt.binary

    def test_datetime_types(self):
        """Test parsing of date/time types."""
        assert parse_db2_type("DATE") == dt.date
        assert parse_db2_type("TIME") == dt.time
        assert parse_db2_type("TIMESTAMP") == dt.timestamp
        assert parse_db2_type("TIMESTAMP(6)") == dt.timestamp

    def test_boolean_type(self):
        """Test parsing of boolean type."""
        assert parse_db2_type("BOOLEAN") == dt.boolean

    def test_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert parse_db2_type("varchar(100)") == dt.string
        assert parse_db2_type("INTEGER") == dt.int32
        assert parse_db2_type("integer") == dt.int32

    def test_unknown_type(self):
        """Test that unknown types default to string."""
        assert parse_db2_type("UNKNOWN_TYPE") == dt.string


class TestIbisTypeToDB2Type:
    """Tests for ibis_type_to_db2_type function."""

    def test_integer_types(self):
        """Test conversion of integer types."""
        assert ibis_type_to_db2_type(dt.int8) == "SMALLINT"
        assert ibis_type_to_db2_type(dt.int16) == "SMALLINT"
        assert ibis_type_to_db2_type(dt.int32) == "INTEGER"
        assert ibis_type_to_db2_type(dt.int64) == "BIGINT"

    def test_unsigned_integer_types(self):
        """Test conversion of unsigned integer types."""
        assert ibis_type_to_db2_type(dt.uint8) == "SMALLINT"
        assert ibis_type_to_db2_type(dt.uint16) == "INTEGER"
        assert ibis_type_to_db2_type(dt.uint32) == "BIGINT"
        assert ibis_type_to_db2_type(dt.uint64) == "BIGINT"

    def test_float_types(self):
        """Test conversion of floating point types."""
        assert ibis_type_to_db2_type(dt.float32) == "REAL"
        assert ibis_type_to_db2_type(dt.float64) == "DOUBLE"

    def test_decimal_types(self):
        """Test conversion of decimal types."""
        assert ibis_type_to_db2_type(dt.Decimal(10, 2)) == "DECIMAL(10, 2)"
        assert ibis_type_to_db2_type(dt.Decimal(15, 3)) == "DECIMAL(15, 3)"
        assert ibis_type_to_db2_type(dt.Decimal()) == "DECIMAL(10, 0)"

    def test_string_type(self):
        """Test conversion of string type."""
        assert ibis_type_to_db2_type(dt.string) == "VARCHAR(32672)"

    def test_binary_type(self):
        """Test conversion of binary type."""
        assert ibis_type_to_db2_type(dt.binary) == "VARBINARY(32672)"

    def test_datetime_types(self):
        """Test conversion of date/time types."""
        assert ibis_type_to_db2_type(dt.date) == "DATE"
        assert ibis_type_to_db2_type(dt.time) == "TIME"
        assert ibis_type_to_db2_type(dt.timestamp) == "TIMESTAMP"

    def test_boolean_type(self):
        """Test conversion of boolean type."""
        assert ibis_type_to_db2_type(dt.boolean) == "BOOLEAN"

    def test_json_type(self):
        """Test conversion of JSON type."""
        assert ibis_type_to_db2_type(dt.json) == "CLOB"

    def test_uuid_type(self):
        """Test conversion of UUID type."""
        assert ibis_type_to_db2_type(dt.uuid) == "CHAR(36)"

    def test_complex_types(self):
        """Test conversion of complex types."""
        # Array, Map, and Struct should be stored as CLOB (JSON)
        assert ibis_type_to_db2_type(dt.Array(dt.int32)) == "CLOB"
        assert ibis_type_to_db2_type(dt.Map(dt.string, dt.int32)) == "CLOB"
        assert ibis_type_to_db2_type(dt.Struct({"a": dt.int32, "b": dt.string})) == "CLOB"


class TestTypeCodeToIbisType:
    """Tests for type_code_to_ibis_type function."""

    def test_date_type(self):
        """Test conversion of DATE type code."""
        assert type_code_to_ibis_type(384) == dt.date

    def test_time_type(self):
        """Test conversion of TIME type code."""
        assert type_code_to_ibis_type(388) == dt.time

    def test_timestamp_type(self):
        """Test conversion of TIMESTAMP type code."""
        assert type_code_to_ibis_type(392) == dt.timestamp

    def test_string_types(self):
        """Test conversion of string type codes."""
        assert type_code_to_ibis_type(448) == dt.string  # VARCHAR
        assert type_code_to_ibis_type(452) == dt.string  # CHAR
        assert type_code_to_ibis_type(460) == dt.string  # CLOB

    def test_binary_types(self):
        """Test conversion of binary type codes."""
        assert type_code_to_ibis_type(464) == dt.binary  # BLOB
        assert type_code_to_ibis_type(468) == dt.binary  # BINARY
        assert type_code_to_ibis_type(472) == dt.binary  # VARBINARY

    def test_numeric_types(self):
        """Test conversion of numeric type codes."""
        assert type_code_to_ibis_type(480) == dt.float64  # FLOAT
        assert type_code_to_ibis_type(484) == dt.float64  # DOUBLE
        assert type_code_to_ibis_type(496) == dt.int32    # INTEGER
        assert type_code_to_ibis_type(500) == dt.int16    # SMALLINT
        assert type_code_to_ibis_type(504) == dt.int64    # BIGINT

    def test_decimal_type_with_precision(self):
        """Test conversion of DECIMAL type code with precision."""
        result = type_code_to_ibis_type(492, precision=10, scale=2)
        assert isinstance(result, dt.Decimal)
        assert result.precision == 10
        assert result.scale == 2

    def test_boolean_type(self):
        """Test conversion of BOOLEAN type code."""
        assert type_code_to_ibis_type(908) == dt.boolean

    def test_unknown_type_code(self):
        """Test that unknown type codes default to string."""
        assert type_code_to_ibis_type(9999) == dt.string


class TestRoundTripConversion:
    """Tests for round-trip type conversions."""

    def test_integer_round_trip(self):
        """Test round-trip conversion of integer types."""
        for ibis_type in [dt.int16, dt.int32, dt.int64]:
            db2_type = ibis_type_to_db2_type(ibis_type)
            result = parse_db2_type(db2_type)
            assert result == ibis_type

    def test_float_round_trip(self):
        """Test round-trip conversion of float types."""
        for ibis_type in [dt.float32, dt.float64]:
            db2_type = ibis_type_to_db2_type(ibis_type)
            result = parse_db2_type(db2_type)
            assert result == ibis_type

    def test_datetime_round_trip(self):
        """Test round-trip conversion of datetime types."""
        for ibis_type in [dt.date, dt.time, dt.timestamp]:
            db2_type = ibis_type_to_db2_type(ibis_type)
            result = parse_db2_type(db2_type)
            assert result == ibis_type

    def test_boolean_round_trip(self):
        """Test round-trip conversion of boolean type."""
        db2_type = ibis_type_to_db2_type(dt.boolean)
        result = parse_db2_type(db2_type)
        assert result == dt.boolean
