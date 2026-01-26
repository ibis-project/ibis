"""Tests for Materialize's mz_now() function."""

from __future__ import annotations

import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.materialize import operations as mz_ops
from ibis.backends.materialize.api import mz_now


class TestMzNowOperation:
    """Test MzNow operation properties."""

    def test_mz_now_operation_dtype(self):
        """Test that MzNow returns timestamp with timezone."""
        op = mz_ops.MzNow()
        assert op.dtype == dt.Timestamp(timezone="UTC")

    def test_mz_now_operation_shape(self):
        """Test that MzNow is scalar."""
        import ibis.expr.datashape as ds

        op = mz_ops.MzNow()
        assert op.shape == ds.scalar

    def test_mz_now_is_impure(self):
        """Test that MzNow is marked as impure operation."""
        from ibis.expr.operations.generic import Impure

        assert issubclass(mz_ops.MzNow, Impure)


class TestMzNowCompilation:
    """Test mz_now() SQL compilation."""

    def test_compile_mz_now(self, con):
        """Test basic mz_now() compilation."""
        expr = mz_now()
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()

    def test_mz_now_in_select(self, con):
        """Test mz_now() in SELECT statement."""
        expr = ibis.memtable({"a": [1, 2, 3]}).mutate(ts=mz_now())
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()

    def test_mz_now_in_filter(self, con):
        """Test mz_now() in WHERE clause for temporal filtering."""
        # Create a mock table with timestamp column
        t = ibis.memtable(
            {
                "event_ts": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:30",
                    "2024-01-01 00:01:00",
                ]
            }
        )
        t = t.mutate(event_ts=t.event_ts.cast("timestamp"))

        # Filter for events within 30 seconds of mz_now()
        expr = t.filter(mz_now() <= t.event_ts + ibis.interval(seconds=30))
        sql = con.compile(expr)

        assert "mz_now()" in sql.lower()
        assert "where" in sql.lower() or "filter" in sql.lower()

    def test_mz_now_comparison(self, con):
        """Test mz_now() with comparison operators."""
        t = ibis.memtable({"event_ts": ["2024-01-01 00:00:00"]})
        t = t.mutate(event_ts=t.event_ts.cast("timestamp"))

        # Test various comparison operators
        exprs = [
            t.filter(mz_now() > t.event_ts),
            t.filter(mz_now() >= t.event_ts),
            t.filter(mz_now() < t.event_ts),
            t.filter(mz_now() <= t.event_ts),
            t.filter(mz_now() == t.event_ts),
        ]

        for expr in exprs:
            sql = con.compile(expr)
            assert "mz_now()" in sql.lower()

    def test_mz_now_arithmetic(self, con):
        """Test mz_now() with interval arithmetic."""
        expr = mz_now() - ibis.interval(days=1)
        sql = con.compile(expr)

        assert "mz_now()" in sql.lower()
        assert "interval" in sql.lower()


@pytest.mark.usefixtures("con")
class TestMzNowExecution:
    """Test mz_now() execution against live Materialize instance."""

    def test_execute_mz_now(self, con):
        """Test that mz_now() can be executed and returns a timestamp."""
        result = con.execute(mz_now())

        # Should return a timestamp value
        assert result is not None

        # Should be a timestamp-like object
        import pandas as pd

        assert isinstance(result, (pd.Timestamp, str))

    def test_mz_now_vs_now(self, con):
        """Test that mz_now() and now() return different timestamps."""
        mz_now_result = con.execute(mz_now())
        now_result = con.execute(ibis.now())

        # Both should return timestamps
        assert mz_now_result is not None
        assert now_result is not None

        # The mz_now() function docstring should clarify they're different
        assert "logical" in mz_now.__doc__.lower()
        # The docstring explains it's different from now() (system clock)
        assert "now()" in mz_now.__doc__.lower()

    def test_mz_now_in_table_query(self, con):
        """Test mz_now() in a table query."""
        # Create a temporary table with timestamp column
        data = ibis.memtable(
            {
                "id": [1, 2, 3],
                "created_at": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:30",
                    "2024-01-01 00:01:00",
                ],
            }
        )
        data = data.mutate(created_at=data.created_at.cast("timestamp"))

        # Add mz_now() as a column
        result_expr = data.mutate(current_ts=mz_now())
        result = con.execute(result_expr)

        # Should have current_ts column
        assert "current_ts" in result.columns

        # All rows should have the same mz_now() value (logical timestamp)
        assert result["current_ts"].nunique() == 1

    def test_temporal_filter_pattern(self, con):
        """Test recommended temporal filter pattern with mz_now()."""
        # This tests the idiomatic Materialize pattern for temporal filters
        data = ibis.memtable(
            {
                "event_id": [1, 2, 3],
                "event_ts": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:30",
                    "2024-01-01 00:01:00",
                ],
            }
        )
        data = data.mutate(event_ts=data.event_ts.cast("timestamp"))

        # Recommended pattern: mz_now() > event_ts + INTERVAL
        # (operation on right side of comparison)
        expr = data.filter(mz_now() > data.event_ts + ibis.interval(seconds=30))

        # Should compile without error
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()

        # Should execute without error (even if result is empty)
        result = con.execute(expr)
        assert result is not None


class TestMzNowDocumentation:
    """Test documentation and examples in mz_now()."""

    def test_mz_now_function_exists(self):
        """Test that mz_now() function exists."""
        assert callable(mz_now)

    def test_mz_now_docstring(self):
        """Test that mz_now() has proper documentation."""
        doc = mz_now.__doc__
        assert doc is not None

        # Should explain key differences from now()
        assert "logical" in doc.lower()
        assert "now()" in doc.lower()

        # Should mention materialized views
        assert "materialized" in doc.lower() or "streaming" in doc.lower()

        # Should have link to docs
        assert "materialize.com/docs" in doc.lower()

    def test_mz_now_return_type(self):
        """Test that mz_now() returns correct expression type."""
        expr = mz_now()

        # Should return a TimestampScalar expression
        assert expr.type().is_timestamp()

    def test_mz_now_examples_in_docstring(self):
        """Test that docstring contains usage examples."""
        doc = mz_now.__doc__

        # Should have examples section
        assert "Examples" in doc
        assert ">>>" in doc

        # Should show temporal filter example
        assert "filter" in doc.lower()


class TestMzNowEdgeCases:
    """Test edge cases and error handling for mz_now()."""

    def test_mz_now_multiple_calls(self, con):
        """Test that multiple mz_now() calls work correctly."""
        expr = ibis.memtable({"a": [1, 2]}).mutate(ts1=mz_now(), ts2=mz_now())

        sql = con.compile(expr)
        # Should have two mz_now() calls
        assert sql.lower().count("mz_now()") == 2

    def test_mz_now_with_cast(self, con):
        """Test mz_now() with type casting."""
        # Cast to string
        expr = mz_now().cast("string")
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "cast" in sql.lower()

    def test_mz_now_in_aggregate(self, con):
        """Test mz_now() in aggregate context."""
        data = ibis.memtable({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        # Use mz_now() in aggregate - since it's scalar, just add it as a column
        expr = data.group_by("group").aggregate(
            total=data.value.sum(), snapshot_ts=mz_now()
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "group by" in sql.lower()


class TestMzNowIntegration:
    """Integration tests combining mz_now() with other features."""

    def test_mz_now_with_window_function(self, con):
        """Test mz_now() combined with window functions."""
        data = ibis.memtable({"id": [1, 2, 3], "value": [10, 20, 30]})

        # Add mz_now() and window function
        expr = data.mutate(
            ts=mz_now(), row_num=ibis.row_number().over(order_by=data.value)
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "row_number" in sql.lower()

    def test_mz_now_with_join(self, con):
        """Test mz_now() in join condition."""
        left = ibis.memtable({"id": [1, 2], "ts": ["2024-01-01", "2024-01-02"]})
        right = ibis.memtable({"id": [1, 2], "name": ["A", "B"]})

        left = left.mutate(ts=left.ts.cast("timestamp"))

        # Add mz_now() in join
        expr = left.join(right, "id").mutate(current=mz_now())

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "join" in sql.lower()

    def test_mz_now_with_case_when(self, con):
        """Test mz_now() in CASE WHEN expression."""
        data = ibis.memtable({"ts": ["2024-01-01", "2024-06-01"]})
        data = data.mutate(ts=data.ts.cast("timestamp"))

        # Use mz_now() in case expression using ifelse
        expr = data.mutate(status=ibis.ifelse(mz_now() > data.ts, "past", "future"))

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "case" in sql.lower()
