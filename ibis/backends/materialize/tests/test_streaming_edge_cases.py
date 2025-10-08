"""Edge case tests for streaming operations in Materialize.

This module tests complex temporal patterns, window functions in streaming contexts,
and join operations with streaming semantics.

References:
- https://materialize.com/docs/transform-data/idiomatic-materialize-sql/
- https://materialize.com/docs/sql/functions/now_and_mz_now/
- Coverage analysis: MATERIALIZE_TEST_COVERAGE_ANALYSIS.md
"""

from __future__ import annotations

import pytest

import ibis
from ibis.backends.materialize.api import mz_now


@pytest.mark.usefixtures("con")
class TestMzNowComplexScenarios:
    """Test mz_now() in complex query patterns (P1 - High Priority)."""

    def test_mz_now_in_multiple_filters(self, con):
        """Test mz_now() used in multiple filter conditions.

        Multiple mz_now() calls in the same query should use the same
        logical timestamp for consistency.
        """
        t = ibis.memtable(
            {
                "id": [1, 2, 3],
                "start_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "end_time": ["2024-01-05", "2024-01-06", "2024-01-07"],
            }
        )
        t = t.mutate(
            start_time=t.start_time.cast("timestamp"),
            end_time=t.end_time.cast("timestamp"),
        )

        # Multiple mz_now() filters
        expr = t.filter((mz_now() > t.start_time) & (mz_now() < t.end_time))

        sql = con.compile(expr)
        # Should have multiple mz_now() calls
        assert sql.lower().count("mz_now()") >= 2

    def test_mz_now_with_case_when_expressions(self, con):
        """Test mz_now() in CASE WHEN expressions.

        Note: Already tested in test_mz_now.py, but adding more edge cases.
        """
        t = ibis.memtable(
            {
                "id": [1, 2, 3],
                "deadline": ["2024-01-01", "2024-06-01", "2024-12-31"],
            }
        )
        t = t.mutate(deadline=t.deadline.cast("timestamp"))

        # Complex CASE with mz_now() using nested ifelse
        expr = t.mutate(
            status=ibis.ifelse(
                mz_now() > t.deadline + ibis.interval(days=30),
                "overdue_long",
                ibis.ifelse(
                    mz_now() > t.deadline,
                    "overdue",
                    ibis.ifelse(
                        mz_now() > t.deadline - ibis.interval(days=7),
                        "due_soon",
                        "active",
                    ),
                ),
            )
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "case" in sql.lower()

    def test_mz_now_in_join_condition(self, con):
        """Test mz_now() used in join conditions.

        Temporal joins are important for streaming queries.
        """
        left = ibis.memtable(
            {
                "id": [1, 2, 3],
                "valid_from": ["2024-01-01", "2024-01-05", "2024-01-10"],
            }
        )
        right = ibis.memtable(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        left = left.mutate(valid_from=left.valid_from.cast("timestamp"))

        # Join with temporal condition
        expr = left.join(right, "id").filter(
            mz_now() > left.valid_from + ibis.interval(days=1)
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "join" in sql.lower()

    def test_mz_now_with_subquery(self, con):
        """Test mz_now() in subquery patterns.

        Note: mz_now() behavior in subqueries may have subtleties around
        logical timestamp consistency.
        """
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "created_at": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ],
            }
        )
        t = t.mutate(created_at=t.created_at.cast("timestamp"))

        # Subquery with mz_now()
        recent = t.filter(mz_now() > t.created_at + ibis.interval(days=1))

        # Use subquery result
        expr = recent.group_by("category").aggregate(total=recent.value.sum())

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()


@pytest.mark.usefixtures("con")
class TestIntervalEdgeCases:
    """Test interval arithmetic edge cases (P2 priority but grouping with mz_now())."""

    def test_interval_zero_duration(self, con):
        """Test interval with zero duration."""
        # Zero interval
        expr = mz_now() + ibis.interval(seconds=0)
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()

    def test_interval_negative_duration(self, con):
        """Test interval with negative duration (going backwards in time)."""
        # Negative interval
        expr = mz_now() - ibis.interval(days=7)
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "interval" in sql.lower()

    def test_interval_mixed_units(self, con):
        """Test interval with mixed time units."""
        # Combined interval: 1 day + 2 hours + 30 minutes
        interval = (
            ibis.interval(days=1) + ibis.interval(hours=2) + ibis.interval(minutes=30)
        )
        expr = mz_now() + interval

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()

    def test_interval_very_large(self, con):
        """Test interval with very large duration."""
        # 10 years into the future
        expr = mz_now() + ibis.interval(years=10)
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()

    def test_multiple_interval_operations(self, con):
        """Test chaining multiple interval operations."""
        t = ibis.memtable(
            {
                "id": [1],
                "created_at": ["2024-01-01"],
            }
        )
        t = t.mutate(created_at=t.created_at.cast("timestamp"))

        # Chain multiple interval operations
        expr = t.mutate(
            plus_1d=t.created_at + ibis.interval(days=1),
            minus_1h=t.created_at - ibis.interval(hours=1),
            complex=(t.created_at + ibis.interval(days=7)) - ibis.interval(hours=6),
        )

        sql = con.compile(expr)
        assert "interval" in sql.lower()


@pytest.mark.usefixtures("con")
class TestWindowFunctionsStreaming:
    """Test window functions in streaming contexts (P1 - High Priority).

    Window functions in streaming databases require special attention:
    - ORDER BY is critical for deterministic results
    - Unbounded windows may have performance implications
    """

    def test_window_function_with_order_by(self, con):
        """Test window function with explicit ORDER BY (recommended)."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "seq": [1, 2, 3, 4],
            }
        )

        # Window with ORDER BY (deterministic)
        expr = t.mutate(
            row_num=ibis.row_number().over(
                ibis.window(group_by="category", order_by="seq")
            )
        )

        result = con.execute(expr)
        assert "row_num" in result.columns

    def test_window_rank_with_ties(self, con):
        """Test RANK() window function with tied values."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "A", "B", "B"],
                "score": [100, 100, 90, 85, 85],
            }
        )

        # RANK handles ties
        expr = t.mutate(
            rank=ibis.rank().over(
                ibis.window(group_by="category", order_by=ibis.desc("score"))
            )
        )

        result = con.execute(expr)
        # First two rows in category A should both have rank 1
        assert "rank" in result.columns

    def test_window_lead_lag_streaming(self, con):
        """Test LEAD/LAG window functions in streaming context."""
        t = ibis.memtable(
            {
                "id": [1, 2, 3, 4],
                "value": [10, 20, 30, 40],
                "ts": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            }
        )
        t = t.mutate(ts=t.ts.cast("timestamp"))

        # LAG to get previous value
        expr = t.mutate(
            prev_value=t.value.lag().over(ibis.window(order_by="ts")),
            next_value=t.value.lead().over(ibis.window(order_by="ts")),
        )

        result = con.execute(expr)
        assert "prev_value" in result.columns
        assert "next_value" in result.columns

    def test_window_aggregate_functions(self, con):
        """Test aggregate window functions (SUM, AVG, etc.)."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "seq": [1, 2, 3, 4],
            }
        )

        # Running sum
        expr = t.mutate(
            running_sum=t.value.sum().over(
                ibis.window(group_by="category", order_by="seq")
            )
        )

        result = con.execute(expr)
        # Category A: [10, 30], Category B: [30, 70]
        assert "running_sum" in result.columns

    def test_window_with_mz_now(self, con):
        """Test window functions combined with mz_now()."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "created_at": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ],
            }
        )
        t = t.mutate(created_at=t.created_at.cast("timestamp"))

        # Window function with mz_now() column
        expr = t.mutate(
            query_time=mz_now(),
            row_num=ibis.row_number().over(
                ibis.window(group_by="category", order_by="created_at")
            ),
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "row_number" in sql.lower()


@pytest.mark.usefixtures("con")
class TestJoinEdgeCases:
    """Test join operations in streaming contexts (P1 - High Priority)."""

    def test_join_with_temporal_filter(self, con):
        """Test join combined with mz_now() temporal filter."""
        orders = ibis.memtable(
            {
                "order_id": [1, 2, 3],
                "customer_id": [101, 102, 101],
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )
        customers = ibis.memtable(
            {
                "customer_id": [101, 102, 103],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        orders = orders.mutate(order_date=orders.order_date.cast("timestamp"))

        # Temporal join: only recent orders
        expr = (
            orders.join(customers, "customer_id")
            .filter(mz_now() > orders.order_date + ibis.interval(hours=1))
            .select(orders.order_id, customers.name, orders.order_date)
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "join" in sql.lower()

    def test_join_with_null_keys(self, con):
        """Test join behavior with NULL keys.

        In SQL, NULL != NULL, so NULL keys don't match.
        """
        left = ibis.memtable(
            {"id": [1, 2, None], "left_val": ["a", "b", "c"]},
            schema={"id": "int64", "left_val": "string"},
        )
        right = ibis.memtable(
            {"id": [1, None, 3], "right_val": ["x", "y", "z"]},
            schema={"id": "int64", "right_val": "string"},
        )

        # Inner join - NULL keys won't match
        expr = left.join(right, "id")

        result = con.execute(expr)
        # Should only match id=1 (NULLs don't match)
        assert len(result) == 1

    def test_left_join_with_nulls(self, con):
        """Test LEFT JOIN preserves left side rows with no match."""
        left = ibis.memtable(
            {"id": [1, 2, 3], "left_val": ["a", "b", "c"]},
        )
        right = ibis.memtable(
            {"id": [1], "right_val": ["x"]},
        )

        # Left join - keeps all left rows
        expr = left.left_join(right, "id")

        result = con.execute(expr)
        # Should have all 3 left rows
        assert len(result) == 3

    def test_self_join_pattern(self, con):
        """Test self-join pattern (joining table to itself)."""
        t = ibis.memtable(
            {
                "id": [1, 2, 3],
                "parent_id": [None, 1, 1],
                "name": ["root", "child1", "child2"],
            },
            schema={"id": "int64", "parent_id": "int64", "name": "string"},
        )

        # Self-join to get parent names
        # Note: Use explicit column selection instead of suffixes parameter
        parents = t.view()
        joined = t.left_join(parents, t.parent_id == parents.id)
        expr = joined.select(child_id=t.id, child_name=t.name, parent_name=parents.name)

        result = con.execute(expr)
        assert len(result) == 3

    def test_multiple_joins(self, con):
        """Test query with multiple joins."""
        orders = ibis.memtable(
            {
                "order_id": [1, 2],
                "customer_id": [101, 102],
                "product_id": [201, 202],
            }
        )
        customers = ibis.memtable(
            {
                "customer_id": [101, 102],
                "cust_name": ["Alice", "Bob"],
            }
        )
        products = ibis.memtable(
            {
                "product_id": [201, 202],
                "prod_name": ["Widget", "Gadget"],
            }
        )

        # Multi-join query
        expr = (
            orders.join(customers, "customer_id")
            .join(products, "product_id")
            .select(orders.order_id, customers.cust_name, products.prod_name)
        )

        result = con.execute(expr)
        assert len(result) == 2

    def test_join_with_complex_condition(self, con):
        """Test join with complex ON conditions."""
        left = ibis.memtable(
            {
                "id": [1, 2, 3],
                "min_val": [10, 20, 30],
                "max_val": [50, 60, 70],
            }
        )
        right = ibis.memtable(
            {
                "id": [1, 2, 3],
                "value": [25, 15, 45],
            }
        )

        # Join where right.value is between left.min_val and left.max_val
        expr = left.join(
            right,
            (left.id == right.id)
            & (right.value >= left.min_val)
            & (right.value <= left.max_val),
        )

        result = con.execute(expr)
        # id=1: 25 is in [10, 50] ✓
        # id=2: 15 is not in [20, 60] ✗
        # id=3: 45 is in [30, 70] ✓
        assert len(result) == 2
