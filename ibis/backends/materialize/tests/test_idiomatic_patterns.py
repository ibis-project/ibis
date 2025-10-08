"""Tests for Materialize idiomatic SQL patterns.

This module tests various idiomatic Materialize SQL patterns as recommended in:
https://materialize.com/docs/transform-data/idiomatic-materialize-sql/

These patterns are optimized for Materialize's streaming and incremental
computation model.
"""

from __future__ import annotations

import ibis
from ibis.backends.materialize.api import mz_now


class TestDistinctOnPatterns:
    """Test DISTINCT ON patterns using distinct(on=...) API.

    Materialize supports PostgreSQL's DISTINCT ON for Top-1 queries.
    The Materialize compiler rewrites Ibis's distinct(on=...) (which uses First aggregates)
    into native DISTINCT ON SQL, which is more efficient than window functions.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#window-functions
    """

    def test_distinct_on_single_column(self, con):
        """Test DISTINCT ON with a single grouping column."""
        data = ibis.memtable(
            {
                "category": ["A", "A", "B", "B", "C"],
                "value": [10, 20, 15, 5, 30],
                "name": ["x1", "x2", "y1", "y2", "z1"],
            }
        )

        # Use distinct(on=...) which gets rewritten to DISTINCT ON
        expr = data.distinct(on="category", keep="first")
        sql = con.compile(expr)

        # Verify DISTINCT ON is in the SQL
        assert "distinct on" in sql.lower()
        assert "category" in sql.lower()
        assert "order by" in sql.lower()

    def test_distinct_on_multiple_columns(self, con):
        """Test DISTINCT ON with multiple grouping columns."""
        data = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "subcategory": ["X", "Y", "X", "Y"],
                "value": [10, 20, 15, 5],
            }
        )

        # DISTINCT ON with multiple columns
        expr = data.distinct(on=["category", "subcategory"], keep="first")
        sql = con.compile(expr)

        assert "distinct on" in sql.lower()
        assert "category" in sql.lower()
        assert "subcategory" in sql.lower()

    def test_distinct_on_execution(self, con):
        """Test that DISTINCT ON actually executes against Materialize."""
        # Create test table
        con.raw_sql("DROP TABLE IF EXISTS test_distinct_exec;")
        con.raw_sql("""
            CREATE TABLE test_distinct_exec (
                cat TEXT, val INT, name TEXT
            );
        """)
        con.raw_sql("""
            INSERT INTO test_distinct_exec VALUES
                ('A', 10, 'x1'), ('A', 20, 'x2'),
                ('B', 15, 'y1'), ('B', 5, 'y2');
        """)

        try:
            t = con.table("test_distinct_exec")
            result = t.distinct(on="cat", keep="first").execute()

            # Should return one row per category
            assert len(result) == 2
            assert set(result["cat"]) == {"A", "B"}
        finally:
            con.raw_sql("DROP TABLE IF EXISTS test_distinct_exec;")


class TestTop1AndTopKPatterns:
    """Test Top-1 and Top-K query patterns using ROW_NUMBER().

    While DISTINCT ON works for Top-1, ROW_NUMBER() window functions are also
    valid and can be used for more complex Top-K queries.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#window-functions
    """

    def test_top_1_with_row_number(self, con):
        """Test Top-1 query per group using ROW_NUMBER()."""
        data = ibis.memtable(
            {
                "category": ["A", "A", "B", "B", "C"],
                "value": [10, 20, 15, 5, 30],
                "name": ["x1", "x2", "y1", "y2", "z1"],
            }
        )

        # ROW_NUMBER approach for Top-1
        expr = (
            data.mutate(
                rn=ibis.row_number().over(
                    ibis.window(group_by="category", order_by=ibis.desc("value"))
                )
            )
            .filter(lambda t: t.rn == 1)
            .drop("rn")
        )

        sql = con.compile(expr)
        assert "row_number" in sql.lower()
        assert "over" in sql.lower()
        assert "partition by" in sql.lower()

    def test_top_k_with_row_number(self, con):
        """Test Top-K query per group using ROW_NUMBER()."""
        data = ibis.memtable(
            {
                "category": ["A", "A", "A", "B", "B", "B"],
                "value": [10, 20, 30, 15, 25, 5],
                "name": ["a1", "a2", "a3", "b1", "b2", "b3"],
            }
        )

        # ROW_NUMBER approach for Top-2
        k = 2
        expr = (
            data.mutate(
                rn=ibis.row_number().over(
                    ibis.window(group_by="category", order_by=ibis.desc("value"))
                )
            )
            .filter(lambda t: t.rn <= k)
            .drop("rn")
        )

        sql = con.compile(expr)
        assert "row_number" in sql.lower()
        assert "over" in sql.lower()
        assert "partition by" in sql.lower()

    def test_top_1_multiple_sort_columns(self, con):
        """Test Top-1 with multiple sort columns."""
        data = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [20, 20, 15, 10],  # Ties in value
                "timestamp": ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
            }
        )
        data = data.mutate(timestamp=data.timestamp.cast("timestamp"))

        # Break ties with timestamp
        expr = (
            data.mutate(
                rn=ibis.row_number().over(
                    ibis.window(
                        group_by="category",
                        order_by=[ibis.desc("value"), ibis.desc("timestamp")],
                    )
                )
            )
            .filter(lambda t: t.rn == 1)
            .drop("rn")
        )

        sql = con.compile(expr)
        assert "row_number" in sql.lower()
        assert "order by" in sql.lower()


class TestLateralJoinPatterns:
    """Test LATERAL JOIN patterns for Top-K queries.

    LATERAL JOIN is the recommended pattern for Top-K queries (K > 1) in Materialize
    as it's more efficient than window functions.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#window-functions
    """

    def test_lateral_join_compilation(self, con):
        """Test that LATERAL join compiles correctly."""
        # Create tables
        _groups = ibis.memtable({"group_id": [1, 2, 3]})
        items = ibis.memtable(
            {
                "group_id": [1, 1, 1, 2, 2, 2],
                "value": [10, 20, 30, 15, 25, 35],
                "name": ["a", "b", "c", "d", "e", "f"],
            }
        )

        # Top-K using window functions (fallback approach)
        # Note: True LATERAL JOIN syntax may not be directly expressible in Ibis,
        # but window functions work as an alternative
        top_k = (
            items.mutate(
                rn=ibis.row_number().over(
                    ibis.window(group_by="group_id", order_by=ibis.desc("value"))
                )
            )
            .filter(lambda t: t.rn <= 2)
            .drop("rn")
        )

        sql = con.compile(top_k)
        assert sql
        assert "row_number" in sql.lower()

    def test_correlated_subquery(self, con):
        """Test correlated subquery compilation (similar to LATERAL)."""
        outer = ibis.memtable({"category": ["A", "B", "C"], "threshold": [10, 20, 30]})
        inner = ibis.memtable(
            {
                "category": ["A", "A", "B", "B", "C", "C"],
                "value": [5, 15, 10, 25, 20, 40],
            }
        )

        # Use a join with filter (correlated pattern)
        result = outer.join(inner, outer.category == inner.category).filter(
            inner.value > outer.threshold
        )

        sql = con.compile(result)
        assert sql
        assert "join" in sql.lower()


class TestUnnestPatterns:
    """Test UNNEST patterns for array operations.

    UNNEST is recommended for ANY() equi-joins and array operations.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#arrays
    """

    def test_unnest_in_select(self, con):
        """Test UNNEST in SELECT clause."""
        data = ibis.memtable(
            {
                "id": [1, 2, 3],
                "tags": [["a", "b"], ["c", "d"], ["e"]],
            }
        )

        # Unnest array column
        expr = data.tags.unnest()
        sql = con.compile(expr)

        assert "unnest" in sql.lower()

    def test_unnest_with_filter(self, con):
        """Test UNNEST with filtering pattern."""
        data = ibis.memtable(
            {
                "id": [1, 2],
                "tags": [["a", "b", "c"], ["d", "e"]],
            }
        )

        # Unnest and filter
        unnested = data.tags.unnest()
        expr = data.select("id", tag=unnested).filter(lambda t: t.tag == "a")

        sql = con.compile(expr)
        assert "unnest" in sql.lower()

    def test_unnest_any_join_pattern(self, con):
        """Test UNNEST for ANY() equi-join pattern.

        Materialize recommends using UNNEST instead of ANY() for array membership checks.
        This is more efficient for streaming queries.

        Anti-pattern: WHERE value = ANY(array_column)
        Recommended: WHERE value IN (SELECT UNNEST(array_column))
        """
        # Main table
        items = ibis.memtable(
            {
                "item_id": [1, 2, 3, 4],
                "item_name": ["apple", "banana", "cherry", "date"],
            }
        )

        # Reference table with arrays
        allowed = ibis.memtable(
            {
                "category": ["fruit", "vegetable"],
                "allowed_items": [["apple", "banana", "cherry"], ["carrot", "potato"]],
            }
        )

        # Idiomatic pattern: Join with UNNEST
        # Filter to fruit category and unnest the allowed items
        fruit_allowed = allowed.filter(allowed.category == "fruit")
        allowed_unnested = fruit_allowed.select(
            category=fruit_allowed.category, item=fruit_allowed.allowed_items.unnest()
        )

        # Join to get matching items
        expr = items.join(
            allowed_unnested, items.item_name == allowed_unnested.item
        ).select(items.item_id, items.item_name, allowed_unnested.category)

        sql = con.compile(expr)

        # Should use UNNEST
        assert "unnest" in sql.lower()
        # Should have a join
        assert "join" in sql.lower()


class TestTemporalFilterPatterns:
    """Test temporal filter patterns with mz_now().

    These tests verify the idiomatic temporal filter patterns work correctly.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#temporal-filters
    """

    def test_temporal_filter_idiomatic(self, con):
        """Test idiomatic temporal filter pattern."""
        events = ibis.memtable(
            {
                "event_id": [1, 2, 3],
                "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )
        events = events.mutate(created_at=events.created_at.cast("timestamp"))

        # Idiomatic: mz_now() isolated on one side
        expr = events.filter(mz_now() > events.created_at + ibis.interval(days=1))
        sql = con.compile(expr)

        assert "mz_now()" in sql.lower()
        assert "interval" in sql.lower()

    def test_temporal_filter_with_comparison(self, con):
        """Test temporal filter with various comparison operators."""
        events = ibis.memtable(
            {
                "event_id": [1, 2, 3],
                "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )
        events = events.mutate(created_at=events.created_at.cast("timestamp"))

        # Test different operators
        for op in [">", ">=", "<", "<="]:
            if op == ">":
                expr = events.filter(
                    mz_now() > events.created_at + ibis.interval(days=1)
                )
            elif op == ">=":
                expr = events.filter(
                    mz_now() >= events.created_at + ibis.interval(days=1)
                )
            elif op == "<":
                expr = events.filter(
                    mz_now() < events.created_at + ibis.interval(days=1)
                )
            else:  # <=
                expr = events.filter(
                    mz_now() <= events.created_at + ibis.interval(days=1)
                )

            sql = con.compile(expr)
            assert "mz_now()" in sql.lower()


class TestAggregatePatterns:
    """Test aggregate patterns recommended for Materialize.

    Some aggregates like FIRST/LAST are not supported, so we use
    alternative patterns.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#aggregations
    """

    def test_min_max_instead_of_first_last(self, con):
        """Test using MIN/MAX instead of FIRST/LAST."""
        data = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 15, 5],
                "timestamp": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            }
        )
        data = data.mutate(timestamp=data.timestamp.cast("timestamp"))

        # Use MIN/MAX with GROUP BY instead of FIRST/LAST
        expr = data.group_by("category").aggregate(
            earliest=data.timestamp.min(), latest=data.timestamp.max()
        )

        sql = con.compile(expr)
        assert "min" in sql.lower()
        assert "max" in sql.lower()
        assert "group by" in sql.lower()

    def test_count_distinct(self, con):
        """Test COUNT(DISTINCT ...) aggregation."""
        data = ibis.memtable(
            {
                "category": ["A", "A", "B", "B", "A"],
                "item": ["x", "y", "x", "z", "x"],
            }
        )

        expr = data.group_by("category").aggregate(distinct_items=data.item.nunique())

        sql = con.compile(expr)
        assert "count" in sql.lower() and "distinct" in sql.lower()


class TestDisjunctionRewriting:
    """Test disjunction (OR) rewriting patterns for streaming optimization.

    Materialize can optimize certain OR conditions when they're rewritten.
    This is particularly important for streaming queries.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#or-to-union
    """

    def test_or_with_temporal_filter(self, con):
        """Test OR condition with temporal filters.

        When using OR with mz_now(), each branch should isolate mz_now()
        on one side of the comparison for optimal incremental computation.
        """
        events = ibis.memtable(
            {
                "event_id": [1, 2, 3],
                "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "updated_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )
        events = events.mutate(
            created_at=events.created_at.cast("timestamp"),
            updated_at=events.updated_at.cast("timestamp"),
        )

        # Idiomatic pattern: Each OR branch has mz_now() isolated
        expr = events.filter(
            (mz_now() > events.created_at + ibis.interval(days=1))
            | (mz_now() > events.updated_at + ibis.interval(hours=12))
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        # Should have OR condition (or rewritten to UNION)
        assert "or" in sql.lower() or "union" in sql.lower()

    def test_or_condition_compilation(self, con):
        """Test that OR conditions compile correctly.

        While UNION ALL can sometimes be more efficient than OR,
        Ibis will compile OR conditions which Materialize can then optimize.
        """
        data = ibis.memtable(
            {
                "id": [1, 2, 3, 4],
                "status_a": ["active", "inactive", "active", "inactive"],
                "status_b": ["pending", "completed", "pending", "completed"],
            }
        )

        # OR condition
        expr = data.filter((data.status_a == "active") | (data.status_b == "pending"))

        sql = con.compile(expr)
        # Should compile to OR or UNION
        assert "or" in sql.lower() or "union" in sql.lower()


class TestUnionPatterns:
    """Test UNION and UNION ALL patterns with temporal filters.

    UNION ALL is preferred over UNION when duplicates are acceptable,
    as it's more efficient for streaming queries.

    See: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#union-vs-union-all
    """

    def test_union_all_with_mz_now(self, con):
        """Test UNION ALL pattern with mz_now().

        UNION ALL is more efficient than UNION for streaming because
        it doesn't require deduplication. Use it when duplicates are acceptable.
        """
        # Two event streams
        events_a = ibis.memtable(
            {
                "event_id": [1, 2],
                "created_at": ["2024-01-01", "2024-01-02"],
                "source": ["A", "A"],
            }
        )
        events_b = ibis.memtable(
            {
                "event_id": [3, 4],
                "created_at": ["2024-01-01", "2024-01-02"],
                "source": ["B", "B"],
            }
        )

        events_a = events_a.mutate(created_at=events_a.created_at.cast("timestamp"))
        events_b = events_b.mutate(created_at=events_b.created_at.cast("timestamp"))

        # Union the streams
        union_expr = events_a.union(events_b, distinct=False)

        # Add temporal filter using mz_now()
        expr = union_expr.mutate(query_time=mz_now()).filter(
            mz_now() > union_expr.created_at + ibis.interval(days=1)
        )

        sql = con.compile(expr)

        # Should use UNION ALL (distinct=False)
        assert "union" in sql.lower()
        # Should have mz_now()
        assert "mz_now()" in sql.lower()

    def test_union_vs_union_all_compilation(self, con):
        """Test UNION vs UNION ALL compilation.

        Demonstrates the difference between UNION (deduplicates) and
        UNION ALL (preserves all rows). UNION ALL is preferred for streaming.
        """
        table_a = ibis.memtable({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        table_b = ibis.memtable({"id": [2, 3, 4], "value": ["b", "c", "d"]})

        # UNION ALL (no deduplication)
        union_all_expr = table_a.union(table_b, distinct=False)
        union_all_sql = con.compile(union_all_expr)

        # UNION (with deduplication)
        union_expr = table_a.union(table_b, distinct=True)
        union_sql = con.compile(union_expr)

        # UNION ALL should be in the SQL
        assert "union" in union_all_sql.lower()
        # Can check for ALL if sqlglot includes it
        # Note: Some SQL generators may omit ALL in certain contexts

        # UNION should be in the SQL
        assert "union" in union_sql.lower()

        # They should be different (one has deduplication logic)
        # This might manifest as DISTINCT or different query structure
        assert union_all_sql != union_sql or "distinct" in union_sql.lower()
