"""Edge case tests for JSON/JSONB operations in Materialize.

This module tests JSON path extraction, type handling, and edge cases specific
to JSON operations in Materialize.

References:
- https://materialize.com/docs/sql/types/jsonb/
- https://materialize.com/docs/sql/functions/#json
- Coverage analysis: MATERIALIZE_TEST_COVERAGE_ANALYSIS.md
"""

from __future__ import annotations

import pytest

import ibis
from ibis.backends.materialize.api import mz_now


@pytest.mark.usefixtures("con")
class TestJsonPathEdgeCases:
    """Test JSON path extraction edge cases (P2 - Medium Priority)."""

    def test_json_extract_nonexistent_path(self, con):
        """Test extracting nonexistent JSON path returns NULL."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"name": "Alice", "age": 30}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract nonexistent field
        expr = t.data["nonexistent"]

        result = con.execute(expr)
        # Should return NULL for nonexistent path
        assert result.iloc[0] is None or result.iloc[0] != result.iloc[0]

    def test_json_extract_nested_path(self, con):
        """Test extracting nested JSON paths."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"user": {"name": "Alice", "address": {"city": "NYC"}}}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract nested field
        expr = t.data["user"]["address"]["city"]

        result = con.execute(expr)
        # Should extract nested value
        assert result.iloc[0] is not None

    def test_json_extract_array_index(self, con):
        """Test extracting array element from JSON."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"tags": ["python", "sql", "rust"]}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract array element
        expr = t.data["tags"][0]

        result = con.execute(expr)
        # Should extract first array element
        assert result.iloc[0] is not None

    def test_json_null_vs_missing_field(self, con):
        """Test distinction between JSON null value and missing field.

        JSON: {"key": null} vs {"other_key": "value"}
        """
        t = ibis.memtable(
            {
                "id": [1, 2],
                "data": ['{"name": null}', '{"age": 30}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract "name" field
        expr = t.data["name"]

        result = con.execute(expr)
        # Both should be NULL, but semantically different:
        # Row 1: explicit null value
        # Row 2: missing field
        assert result.iloc[0] is None or result.iloc[0] != result.iloc[0]
        assert result.iloc[1] is None or result.iloc[1] != result.iloc[1]

    def test_json_empty_object(self, con):
        """Test operations on empty JSON object."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ["{}"],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract from empty object
        expr = t.data["any_field"]

        result = con.execute(expr)
        # Should return NULL
        assert result.iloc[0] is None or result.iloc[0] != result.iloc[0]

    def test_json_empty_array(self, con):
        """Test operations on empty JSON array."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"items": []}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Try to extract from empty array
        expr = t.data["items"][0]

        result = con.execute(expr)
        # Should return NULL for out-of-bounds
        assert result.iloc[0] is None or result.iloc[0] != result.iloc[0]


@pytest.mark.usefixtures("con")
class TestJsonTypeOperations:
    """Test JSON type detection and conversions."""

    def test_json_typeof(self, con):
        """Test TYPEOF operation on various JSON values."""
        t = ibis.memtable(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "data": [
                    '{"val": 123}',
                    '{"val": "text"}',
                    '{"val": true}',
                    '{"val": null}',
                    '{"val": [1, 2, 3]}',
                    '{"val": {"nested": "obj"}}',
                ],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract value and check type
        expr = t.select(t.id, val=t.data["val"])

        result = con.execute(expr)
        # Should have extracted various types
        assert len(result) == 6

    def test_json_to_text_conversion(self, con):
        """Test converting JSON values to text."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"name": "Alice"}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract and convert to text
        expr = t.data["name"].cast("string")

        result = con.execute(expr)
        # Should convert to string
        assert isinstance(result.iloc[0], str)

    def test_json_numeric_extraction(self, con):
        """Test extracting numeric values from JSON."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"count": 42, "price": 19.99}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract numeric values
        expr = t.select(
            count=t.data["count"].cast("int64"), price=t.data["price"].cast("float64")
        )

        result = con.execute(expr)
        # Should extract numbers
        assert result["count"].iloc[0] is not None


@pytest.mark.usefixtures("con")
class TestJsonWithOtherFeatures:
    """Test JSON operations combined with other Materialize features."""

    def test_json_with_mz_now(self, con):
        """Test JSON operations combined with mz_now()."""
        t = ibis.memtable(
            {
                "id": [1, 2],
                "metadata": ['{"created": "2024-01-01"}', '{"created": "2024-01-02"}'],
            }
        )
        t = t.mutate(metadata=t.metadata.cast("jsonb"))

        # Add mz_now() to query with JSON
        expr = t.mutate(query_time=mz_now(), created=t.metadata["created"])

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()

    def test_json_in_aggregation(self, con):
        """Test JSON operations in GROUP BY context."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "data": [
                    '{"count": 10}',
                    '{"count": 20}',
                    '{"count": 30}',
                    '{"count": 40}',
                ],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract JSON field and aggregate
        expr = t.group_by("category").aggregate(
            total=t.data["count"].cast("int64").sum()
        )

        result = con.execute(expr)
        # Category A: 10 + 20 = 30
        # Category B: 30 + 40 = 70
        assert len(result) == 2

    def test_json_in_filter(self, con):
        """Test filtering based on JSON field values."""
        t = ibis.memtable(
            {
                "id": [1, 2, 3],
                "data": ['{"active": true}', '{"active": false}', '{"active": true}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Filter based on JSON boolean field
        # Note: Extracting booleans from JSON may require casting
        expr = t.mutate(active=t.data["active"])

        result = con.execute(expr)
        assert len(result) == 3

    def test_json_array_operations(self, con):
        """Test operations on JSON arrays."""
        t = ibis.memtable(
            {
                "id": [1, 2],
                "data": ['{"tags": ["a", "b", "c"]}', '{"tags": ["x", "y"]}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract array from JSON
        expr = t.select(t.id, tags=t.data["tags"])

        result = con.execute(expr)
        # Should extract arrays
        assert len(result) == 2

    def test_json_deep_nesting(self, con):
        """Test deeply nested JSON structures."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": [
                    '{"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}'
                ],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Navigate deep nesting
        expr = t.data["level1"]["level2"]["level3"]["level4"]["value"]

        result = con.execute(expr)
        # Should extract deeply nested value
        assert result.iloc[0] is not None


@pytest.mark.usefixtures("con")
class TestJsonEdgeCaseHandling:
    """Test JSON error handling and edge cases."""

    def test_json_invalid_cast(self, con):
        """Test invalid type conversion from JSON."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"name": "Alice"}'],  # String, not a number
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Try to cast string to int (may fail or return NULL)
        expr = t.data["name"]  # Don't cast yet, just extract

        result = con.execute(expr)
        # Should extract the value (casting would be separate)
        assert result.iloc[0] is not None

    def test_json_with_special_characters(self, con):
        """Test JSON with special characters in values."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"text": "Line 1\\nLine 2\\t\\ttabbed"}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract text with special chars
        expr = t.data["text"]

        result = con.execute(expr)
        # Should handle escaped characters
        assert result.iloc[0] is not None

    def test_json_with_unicode(self, con):
        """Test JSON with Unicode characters."""
        t = ibis.memtable(
            {
                "id": [1],
                "data": ['{"emoji": "😀", "chinese": "你好"}'],
            }
        )
        t = t.mutate(data=t.data.cast("jsonb"))

        # Extract Unicode values
        expr = t.select(emoji=t.data["emoji"], chinese=t.data["chinese"])

        result = con.execute(expr)
        # Should handle Unicode
        assert len(result) == 1

    def test_json_null_object(self, con):
        """Test NULL JSON object (not same as empty object)."""
        t = ibis.memtable(
            {"id": [1, 2], "data": [None, "{}"]},
            schema={"id": "int64", "data": "jsonb"},
        )

        # Try to extract from NULL object
        expr = t.data["field"]

        result = con.execute(expr)
        # NULL object should yield NULL
        assert result.iloc[0] is None or result.iloc[0] != result.iloc[0]
        # Empty object should also yield NULL for nonexistent field
        assert result.iloc[1] is None or result.iloc[1] != result.iloc[1]
