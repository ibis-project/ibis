"""Tests for SingleStoreDB SQL compiler type casting and operations."""

from __future__ import annotations

import pytest
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.singlestoredb import SingleStoreDBCompiler


@pytest.fixture
def compiler():
    """Create a SingleStoreDB compiler instance."""
    return SingleStoreDBCompiler()


class TestSingleStoreDBCompiler:
    """Test SingleStoreDB SQL compiler functionality."""

    def test_compiler_uses_singlestoredb_type_mapper(self, compiler):
        """Test that the compiler uses SingleStoreDB type mapper."""
        from ibis.backends.singlestoredb.datatypes import SingleStoreDBType

        assert compiler.type_mapper == SingleStoreDBType

    def test_cast_json_to_json(self, compiler):
        """Test casting JSON to JSON returns the argument unchanged."""
        # Create a mock cast operation
        arg = sge.Column(this="json_col")
        json_type = dt.JSON()

        # Mock the cast operation
        class MockCastOp:
            def __init__(self):
                self.arg = type("MockArg", (), {"dtype": dt.JSON()})()
                self.to = json_type

        op = MockCastOp()
        result = compiler.visit_Cast(op, arg=arg, to=json_type)

        # Should return the original argument for JSON to JSON cast
        assert result == arg

    def test_cast_string_to_json(self, compiler):
        """Test casting string to JSON creates proper CAST expression."""
        arg = sge.Column(this="string_col")
        json_type = dt.JSON()

        class MockCastOp:
            def __init__(self):
                self.arg = type("MockArg", (), {"dtype": dt.String()})()
                self.to = json_type

        op = MockCastOp()
        result = compiler.visit_Cast(op, arg=arg, to=json_type)

        # Should create a CAST expression to JSON
        assert isinstance(result, sge.Cast)
        assert result.to.this == sge.DataType.Type.JSON

    def test_cast_numeric_to_timestamp(self, compiler):
        """Test casting numeric to timestamp handles zero values."""
        arg = sge.Column(this="unix_time")
        timestamp_type = dt.Timestamp()

        class MockCastOp:
            def __init__(self):
                self.arg = type("MockArg", (), {"dtype": dt.Int64()})()
                self.to = timestamp_type

        op = MockCastOp()
        result = compiler.visit_Cast(op, arg=arg, to=timestamp_type)

        # Should use IF statement to handle zero values
        assert isinstance(result, sge.If)

    def test_cast_string_to_binary(self, compiler):
        """Test casting string to binary uses UNHEX function."""
        arg = sge.Column(this="hex_string")
        binary_type = dt.Binary()

        class MockCastOp:
            def __init__(self):
                self.arg = type("MockArg", (), {"dtype": dt.String()})()
                self.to = binary_type

        op = MockCastOp()
        result = compiler.visit_Cast(op, arg=arg, to=binary_type)

        # Should use UNHEX function for string to binary
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "unhex"

    def test_cast_binary_to_string(self, compiler):
        """Test casting binary to string uses HEX function."""
        arg = sge.Column(this="binary_col")
        string_type = dt.String()

        class MockCastOp:
            def __init__(self):
                self.arg = type("MockArg", (), {"dtype": dt.Binary()})()
                self.to = string_type

        op = MockCastOp()
        result = compiler.visit_Cast(op, arg=arg, to=string_type)

        # Should use HEX function for binary to string
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "hex"

    def test_cast_to_geometry(self, compiler):
        """Test casting to geometry type uses ST_GEOMFROMTEXT."""
        arg = sge.Column(this="wkt_string")
        geometry_type = dt.Geometry()

        class MockCastOp:
            def __init__(self):
                self.arg = type("MockArg", (), {"dtype": dt.String()})()
                self.to = geometry_type

        op = MockCastOp()
        result = compiler.visit_Cast(op, arg=arg, to=geometry_type)

        # Should use ST_GEOMFROMTEXT function
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "st_geomfromtext"

    def test_cast_geometry_to_string(self, compiler):
        """Test casting geometry to string uses ST_ASTEXT."""
        arg = sge.Column(this="geom_col")
        string_type = dt.String()

        class MockCastOp:
            def __init__(self):
                self.arg = type("MockArg", (), {"dtype": dt.Geometry()})()
                self.to = string_type

        op = MockCastOp()
        result = compiler.visit_Cast(op, arg=arg, to=string_type)

        # Should use ST_ASTEXT function
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "st_astext"

    def test_nan_not_supported(self, compiler):
        """Test that NaN is not supported in SingleStoreDB."""
        with pytest.raises(NotImplementedError, match="does not support NaN"):
            _ = compiler.NAN

    def test_infinity_not_supported(self, compiler):
        """Test that Infinity is not supported in SingleStoreDB."""
        with pytest.raises(NotImplementedError, match="does not support Infinity"):
            _ = compiler.POS_INF

        with pytest.raises(NotImplementedError, match="does not support Infinity"):
            _ = compiler.NEG_INF

    def test_visit_nonull_literal_decimal_nan_fails(self, compiler):
        """Test that non-finite decimal literals are rejected."""
        import decimal

        class MockOp:
            pass

        op = MockOp()
        nan_decimal = decimal.Decimal("nan")
        decimal_dtype = dt.Decimal(precision=10, scale=2)

        with pytest.raises(com.UnsupportedOperationError):
            compiler.visit_NonNullLiteral(op, value=nan_decimal, dtype=decimal_dtype)

    def test_visit_nonull_literal_binary(self, compiler):
        """Test binary literal handling."""

        class MockOp:
            pass

        op = MockOp()
        binary_value = b"test_data"
        binary_dtype = dt.Binary()

        result = compiler.visit_NonNullLiteral(
            op, value=binary_value, dtype=binary_dtype
        )

        # Should use UNHEX function with hex representation
        assert isinstance(result, sge.Unhex)
        # Verify the hex data is correct
        hex_expected = binary_value.hex()
        assert result.this.this == hex_expected

    def test_visit_nonull_literal_date(self, compiler):
        """Test date literal handling."""
        import datetime

        class MockOp:
            pass

        op = MockOp()
        date_value = datetime.date(2023, 12, 25)
        date_dtype = dt.Date()

        result = compiler.visit_NonNullLiteral(op, value=date_value, dtype=date_dtype)

        # Should use DATE function
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "date"

    def test_visit_nonull_literal_timestamp(self, compiler):
        """Test timestamp literal handling."""
        import datetime

        class MockOp:
            pass

        op = MockOp()
        timestamp_value = datetime.datetime(2023, 12, 25, 10, 30, 45)
        timestamp_dtype = dt.Timestamp()

        result = compiler.visit_NonNullLiteral(
            op, value=timestamp_value, dtype=timestamp_dtype
        )

        # Should use TIMESTAMP function
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "timestamp"

    def test_visit_nonull_literal_time(self, compiler):
        """Test time literal handling."""
        import datetime

        class MockOp:
            pass

        op = MockOp()
        time_value = datetime.time(14, 30, 45, 123456)  # With microseconds
        time_dtype = dt.Time()

        result = compiler.visit_NonNullLiteral(op, value=time_value, dtype=time_dtype)

        # Should use MAKETIME function
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "maketime"

    def test_visit_nonull_literal_unsupported_types(self, compiler):
        """Test that arrays, structs, and maps are unsupported."""

        class MockOp:
            pass

        op = MockOp()

        # Test array type
        array_dtype = dt.Array(dt.int32)
        with pytest.raises(com.UnsupportedBackendType):
            compiler.visit_NonNullLiteral(op, value=[], dtype=array_dtype)

        # Test struct type
        struct_dtype = dt.Struct({"field": dt.string})
        with pytest.raises(com.UnsupportedBackendType):
            compiler.visit_NonNullLiteral(op, value={}, dtype=struct_dtype)

        # Test map type
        map_dtype = dt.Map(dt.string, dt.int32)
        with pytest.raises(com.UnsupportedBackendType):
            compiler.visit_NonNullLiteral(op, value={}, dtype=map_dtype)

    def test_json_get_item_integer_index(self, compiler):
        """Test JSON path extraction with integer index."""

        class MockOp:
            def __init__(self):
                self.index = type("MockIndex", (), {"dtype": dt.Int32()})()

        op = MockOp()
        arg = sge.Column(this="json_col")
        index = sge.Literal.number("0")

        result = compiler.visit_JSONGetItem(op, arg=arg, index=index)

        # Should use JSON_EXTRACT with array index path
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "json_extract"

    def test_json_get_item_string_index(self, compiler):
        """Test JSON path extraction with string key."""

        class MockOp:
            def __init__(self):
                self.index = type("MockIndex", (), {"dtype": dt.String()})()

        op = MockOp()
        arg = sge.Column(this="json_col")
        index = sge.Literal.string("key")

        result = compiler.visit_JSONGetItem(op, arg=arg, index=index)

        # Should use JSON_EXTRACT with object key path
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "json_extract"

    def test_string_find_operation(self, compiler):
        """Test string find operation."""

        class MockOp:
            pass

        op = MockOp()
        arg = sge.Column(this="text_col")
        substr = sge.Literal.string("pattern")
        start = sge.Literal.number("5")

        result = compiler.visit_StringFind(
            op, arg=arg, substr=substr, start=start, end=None
        )

        # Should use LOCATE function with start position
        assert isinstance(result, sge.Anonymous)
        assert result.this.lower() == "locate"

    def test_string_find_with_end_not_supported(self, compiler):
        """Test that string find with end parameter is not supported."""

        class MockOp:
            pass

        op = MockOp()
        arg = sge.Column(this="text_col")
        substr = sge.Literal.string("pattern")
        start = sge.Literal.number("5")
        end = sge.Literal.number("10")

        with pytest.raises(
            NotImplementedError, match="`end` argument is not implemented"
        ):
            compiler.visit_StringFind(op, arg=arg, substr=substr, start=start, end=end)

    def test_minimize_spec_for_rank_operations(self, compiler):
        """Test window spec minimization for rank operations."""

        # Test with rank operation
        class RankOp:
            func = ops.MinRank()  # Use MinRank which inherits from RankBase

        rank_op = RankOp()
        spec = sge.Window()
        result = compiler._minimize_spec(rank_op, spec)
        assert result is None

        # Test with non-rank operation
        class MockSumFunc:
            pass  # Simple mock that's not a RankBase

        class NonRankOp:
            func = MockSumFunc()  # Not a rank operation

        non_rank_op = NonRankOp()
        result = compiler._minimize_spec(non_rank_op, spec)
        assert result == spec


class TestSingleStoreDBCompilerIntegration:
    """Integration tests for the SingleStoreDB compiler."""

    def test_unsupported_operations_inherited_from_mysql(self, compiler):
        """Test that unsupported operations include MySQL unsupported ops."""
        from ibis.backends.sql.compilers.mysql import MySQLCompiler

        # SingleStoreDB should inherit MySQL unsupported operations
        mysql_unsupported = MySQLCompiler.UNSUPPORTED_OPS
        singlestore_unsupported = compiler.UNSUPPORTED_OPS

        # All MySQL unsupported ops should be in SingleStoreDB unsupported ops
        for op in mysql_unsupported:
            assert op in singlestore_unsupported

    def test_simple_ops_inherit_from_mysql(self, compiler):
        """Test that simple operations inherit from MySQL compiler."""
        from ibis.backends.sql.compilers.mysql import MySQLCompiler

        # Should include all MySQL simple operations
        mysql_simple_ops = MySQLCompiler.SIMPLE_OPS
        singlestore_simple_ops = compiler.SIMPLE_OPS

        for op, func_name in mysql_simple_ops.items():
            assert op in singlestore_simple_ops
            assert singlestore_simple_ops[op] == func_name

    def test_rewrites_include_mysql_rewrites(self, compiler):
        """Test that compiler rewrites include MySQL rewrites."""
        from ibis.backends.sql.compilers.mysql import MySQLCompiler

        mysql_rewrites = MySQLCompiler.rewrites
        singlestore_rewrites = compiler.rewrites

        # SingleStoreDB rewrites should include MySQL rewrites
        for rewrite in mysql_rewrites:
            assert rewrite in singlestore_rewrites

    def test_placeholder_distributed_query_methods(self, compiler):
        """Test distributed query optimization methods."""
        query = sge.Select()

        # Test shard key hint method (placeholder)
        result = compiler._add_shard_key_hint(query)
        assert result == query  # Should return unchanged for now

        # Test columnstore optimization method
        result = compiler._optimize_for_columnstore(query)
        # Should add columnstore hint for SELECT queries
        expected = "SELECT /*+ USE_COLUMNSTORE_STRATEGY */"
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__])
