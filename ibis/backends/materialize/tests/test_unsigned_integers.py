"""Tests for Materialize unsigned integer type support.

Materialize supports unsigned integers with byte-count-based naming:
- uint2 (2 bytes, 16 bits) for UInt16
- uint4 (4 bytes, 32 bits) for UInt32
- uint8 (8 bytes, 64 bits) for UInt64

NOTE: Materialize does NOT support 1-byte (8-bit) unsigned integers.

Ref: https://materialize.com/docs/sql/types/uint/
"""

from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt


class TestUnsignedIntegerTypes:
    """Test Materialize's unsigned integer type handling."""

    def test_supported_unsigned_types(self, con):
        """Test that uint16, uint32, and uint64 work correctly."""
        table_name = ibis.util.gen_name("test_uint_table")

        # Create table with supported unsigned integer types
        schema = ibis.schema(
            {
                "id": "int32",
                "u16": "uint16",  # Should map to uint2
                "u32": "uint32",  # Should map to uint4
                "u64": "uint64",  # Should map to uint8
            }
        )

        con.create_table(table_name, schema=schema, overwrite=True)

        try:
            # Verify table exists
            assert table_name in con.list_tables()

            # Get the table and inspect schema
            table = con.table(table_name)
            result_schema = table.schema()

            # Verify types round-trip correctly
            assert result_schema["id"] == dt.Int32()
            assert result_schema["u16"] == dt.UInt16()
            assert result_schema["u32"] == dt.UInt32()
            assert result_schema["u64"] == dt.UInt64()

            # Insert test data using raw SQL (to avoid transaction block issues)
            # Ref: https://materialize.com/docs/sql/begin/
            con.raw_sql(
                f"""
                INSERT INTO {table_name} (id, u16, u32, u64)
                VALUES (1, 100, 1000, 10000),
                       (2, 200, 2000, 20000),
                       (3, 65535, 4294967295, 18446744073709551615)
                """
            )

            # Query back and verify
            result = table.execute()
            assert len(result) == 3
            assert result["u16"].tolist() == [100, 200, 65535]
            assert result["u32"].tolist() == [1000, 2000, 4294967295]

        finally:
            # Clean up
            con.drop_table(table_name, force=True)

    def test_uint8_not_supported(self, con):
        """Test that uint8 (1-byte unsigned int) raises appropriate error."""
        table_name = ibis.util.gen_name("test_uint8_table")

        # Attempting to create a table with uint8 should fail
        schema = ibis.schema({"id": "int32", "u8": "uint8"})

        with pytest.raises(
            com.UnsupportedBackendType,
            match="Materialize doesn't support 1-byte unsigned integers",
        ):
            con.create_table(table_name, schema=schema, overwrite=True)

        # Ensure table was not created
        assert table_name not in con.list_tables()

    def test_unsigned_type_sql_generation(self):
        """Test that correct SQL type names are generated."""
        from ibis.backends.sql.datatypes import MaterializeType

        # Test Ibis → Materialize SQL type mapping
        assert MaterializeType.from_ibis(dt.UInt16()).sql("postgres") == "uint2"
        assert MaterializeType.from_ibis(dt.UInt32()).sql("postgres") == "uint4"
        assert MaterializeType.from_ibis(dt.UInt64()).sql("postgres") == "uint8"

    def test_unsigned_type_parsing(self):
        """Test that Materialize SQL types parse back to correct Ibis types."""
        from ibis.backends.sql.datatypes import MaterializeType

        # Test Materialize SQL → Ibis type parsing
        assert MaterializeType.from_string("uint2") == dt.UInt16()
        assert MaterializeType.from_string("UINT2") == dt.UInt16()  # Case insensitive
        assert MaterializeType.from_string("uint4") == dt.UInt32()
        assert MaterializeType.from_string("uint8") == dt.UInt64()

    def test_unsigned_range_values(self, con):
        """Test boundary values for unsigned integer types."""
        table_name = ibis.util.gen_name("test_uint_ranges")

        schema = ibis.schema(
            {
                "u16_min": "uint16",
                "u16_max": "uint16",
                "u32_min": "uint32",
                "u32_max": "uint32",
                "u64_min": "uint64",
                "u64_max": "uint64",
            }
        )

        con.create_table(table_name, schema=schema, overwrite=True)

        try:
            # Insert boundary values using raw SQL (to avoid transaction block issues)
            # Ref: https://materialize.com/docs/sql/begin/
            con.raw_sql(
                f"""
                INSERT INTO {table_name} (u16_min, u16_max, u32_min, u32_max, u64_min, u64_max)
                VALUES (0, 65535, 0, 4294967295, 0, 18446744073709551615)
                """
            )

            # Query back and verify
            table = con.table(table_name)
            result = table.execute()

            assert result["u16_min"].iloc[0] == 0
            assert result["u16_max"].iloc[0] == 65535
            assert result["u32_min"].iloc[0] == 0
            assert result["u32_max"].iloc[0] == 4294967295
            assert result["u64_min"].iloc[0] == 0
            assert result["u64_max"].iloc[0] == 18446744073709551615

        finally:
            con.drop_table(table_name, force=True)

    def test_unsigned_nullable_vs_non_nullable(self, con):
        """Test nullable and non-nullable unsigned integer columns."""
        table_name = ibis.util.gen_name("test_uint_nullable")

        # Create table with nullable and non-nullable columns
        schema = ibis.schema(
            {
                "nullable_u16": dt.UInt16(nullable=True),
                "non_nullable_u16": dt.UInt16(nullable=False),
            }
        )

        con.create_table(table_name, schema=schema, overwrite=True)

        try:
            # Insert data with NULL using raw SQL (to avoid transaction block issues)
            # Ref: https://materialize.com/docs/sql/begin/
            con.raw_sql(
                f"""
                INSERT INTO {table_name} (nullable_u16, non_nullable_u16)
                VALUES (100, 100),
                       (NULL, 200),
                       (300, 300)
                """
            )

            # Query back
            table = con.table(table_name)
            result = table.execute()

            # Verify nullable column has NULL
            assert result["nullable_u16"].isna().sum() == 1
            # Verify non-nullable column has no NULLs
            assert result["non_nullable_u16"].isna().sum() == 0

        finally:
            con.drop_table(table_name, force=True)
