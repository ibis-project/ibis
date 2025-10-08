"""Edge case tests for aggregations in Materialize.

This module tests edge cases for aggregations, particularly around DISTINCT ON
patterns (Materialize's replacement for First/Last aggregates) and other
aggregate functions in streaming contexts.

References:
- https://materialize.com/docs/transform-data/idiomatic-materialize-sql/
- Coverage analysis: MATERIALIZE_TEST_COVERAGE_ANALYSIS.md
"""

from __future__ import annotations

import pytest

import ibis
from ibis.backends.materialize.api import mz_now


@pytest.mark.usefixtures("con")
class TestDistinctOnEdgeCases:
    """Test edge cases for DISTINCT ON (First aggregate replacement)."""

    def test_distinct_on_with_null_order_by_values(self, con):
        """Test DISTINCT ON when ORDER BY column contains NULLs.

        NULLs in ORDER BY columns can affect which row is selected as "first".
        """
        # Create test data with NULL timestamps
        con.raw_sql("DROP TABLE IF EXISTS test_distinct_nulls;")
        con.raw_sql("""
            CREATE TABLE test_distinct_nulls (
                category TEXT,
                value INT,
                ts TIMESTAMP
            );
        """)
        con.raw_sql("""
            INSERT INTO test_distinct_nulls VALUES
                ('A', 10, '2024-01-01'::TIMESTAMP),
                ('A', 20, NULL),
                ('B', 15, NULL),
                ('B', 25, '2024-01-02'::TIMESTAMP);
        """)

        try:
            t = con.table("test_distinct_nulls")

            # DISTINCT ON with NULL values in order column
            # NULLs sort first/last depending on NULLS FIRST/LAST
            expr = t.order_by(ibis.desc(t.ts)).distinct(on="category", keep="first")

            result = con.execute(expr)

            # Should get one row per category
            assert len(result) == 2
            assert set(result["category"]) == {"A", "B"}

        finally:
            con.raw_sql("DROP TABLE IF EXISTS test_distinct_nulls;")

    def test_distinct_on_with_multiple_order_by_columns(self, con):
        """Test DISTINCT ON with multiple ORDER BY columns for tie-breaking."""
        con.raw_sql("DROP TABLE IF EXISTS test_distinct_multi_order;")
        con.raw_sql("""
            CREATE TABLE test_distinct_multi_order (
                category TEXT,
                value INT,
                priority INT,
                name TEXT
            );
        """)
        con.raw_sql("""
            INSERT INTO test_distinct_multi_order VALUES
                ('A', 100, 1, 'first'),
                ('A', 100, 2, 'second'),
                ('B', 200, 1, 'third'),
                ('B', 200, 1, 'fourth');
        """)

        try:
            t = con.table("test_distinct_multi_order")

            # DISTINCT ON with multiple sort columns
            expr = t.order_by([ibis.desc(t.value), t.priority]).distinct(
                on="category", keep="first"
            )

            result = con.execute(expr)

            assert len(result) == 2
            # Category A should get 'first' (value=100, priority=1)
            # Category B should get 'third' or 'fourth' (value=200, priority=1, same)
            categories = set(result["category"])
            assert categories == {"A", "B"}

        finally:
            con.raw_sql("DROP TABLE IF EXISTS test_distinct_multi_order;")

    def test_distinct_on_empty_table(self, con):
        """Test DISTINCT ON on empty table."""
        con.raw_sql("DROP TABLE IF EXISTS test_distinct_empty;")
        con.raw_sql("""
            CREATE TABLE test_distinct_empty (
                category TEXT,
                value INT
            );
        """)

        try:
            t = con.table("test_distinct_empty")
            expr = t.distinct(on="category", keep="first")
            result = con.execute(expr)

            # Should return empty result
            assert len(result) == 0

        finally:
            con.raw_sql("DROP TABLE IF EXISTS test_distinct_empty;")

    def test_distinct_on_all_same_category(self, con):
        """Test DISTINCT ON when all rows have same grouping value."""
        con.raw_sql("DROP TABLE IF EXISTS test_distinct_same;")
        con.raw_sql("""
            CREATE TABLE test_distinct_same (
                category TEXT,
                value INT,
                name TEXT
            );
        """)
        con.raw_sql("""
            INSERT INTO test_distinct_same VALUES
                ('A', 10, 'first'),
                ('A', 20, 'second'),
                ('A', 30, 'third');
        """)

        try:
            t = con.table("test_distinct_same")
            expr = t.order_by(ibis.desc(t.value)).distinct(on="category", keep="first")
            result = con.execute(expr)

            # Should return just one row (the one with highest value)
            assert len(result) == 1
            assert result["category"].iloc[0] == "A"

        finally:
            con.raw_sql("DROP TABLE IF EXISTS test_distinct_same;")


@pytest.mark.usefixtures("con")
class TestAggregateWithEmptyGroups:
    """Test aggregates with empty groups or no matching rows."""

    def test_aggregate_on_empty_table(self, con):
        """Test aggregation on empty table."""
        con.raw_sql("DROP TABLE IF EXISTS test_agg_empty;")
        con.raw_sql("""
            CREATE TABLE test_agg_empty (
                category TEXT,
                value INT
            );
        """)

        try:
            t = con.table("test_agg_empty")
            expr = t.group_by("category").aggregate(
                total=t.value.sum(), avg=t.value.mean(), cnt=t.value.count()
            )
            result = con.execute(expr)

            # Should return empty result (no groups)
            assert len(result) == 0

        finally:
            con.raw_sql("DROP TABLE IF EXISTS test_agg_empty;")

    def test_aggregate_with_no_matches_after_filter(self, con):
        """Test GROUP BY aggregate when filter eliminates all rows."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
            }
        )

        # Filter that matches nothing
        filtered = t.filter(t.value > 1000)
        expr = filtered.group_by("category").aggregate(total=filtered.value.sum())

        result = con.execute(expr)
        # Should return empty (no groups after filter)
        assert len(result) == 0

    def test_count_on_empty_group(self, con):
        """Test COUNT behavior on empty group."""
        # This is more of a documentation test
        t = ibis.memtable({"category": ["A"], "value": [10]})

        # Group by category and count
        expr = t.group_by("category").aggregate(cnt=t.value.count())

        result = con.execute(expr)
        assert result["cnt"].iloc[0] == 1


@pytest.mark.usefixtures("con")
class TestAggregateWithAllNulls:
    """Test aggregate functions with all-NULL inputs."""

    def test_sum_all_nulls(self, con):
        """Test SUM with all NULL values returns NULL."""
        t = ibis.memtable(
            {"category": ["A", "A"], "value": [None, None]},
            schema={"category": "string", "value": "int64"},
        )

        expr = t.group_by("category").aggregate(total=t.value.sum())
        result = con.execute(expr)

        # SUM of all NULLs should be NULL
        assert result["total"].iloc[0] is None or result["total"].isna().iloc[0]

    def test_count_all_nulls(self, con):
        """Test COUNT with all NULL values returns 0."""
        t = ibis.memtable(
            {"category": ["A", "A"], "value": [None, None]},
            schema={"category": "string", "value": "int64"},
        )

        expr = t.group_by("category").aggregate(cnt=t.value.count())
        result = con.execute(expr)

        # COUNT of all NULLs should be 0 (COUNT ignores NULLs)
        assert result["cnt"].iloc[0] == 0

    def test_avg_all_nulls(self, con):
        """Test AVG with all NULL values returns NULL."""
        t = ibis.memtable(
            {"category": ["A", "A"], "value": [None, None]},
            schema={"category": "string", "value": "int64"},
        )

        expr = t.group_by("category").aggregate(avg=t.value.mean())
        result = con.execute(expr)

        # AVG of all NULLs should be NULL
        assert result["avg"].iloc[0] is None or result["avg"].isna().iloc[0]

    def test_min_max_all_nulls(self, con):
        """Test MIN/MAX with all NULL values return NULL."""
        t = ibis.memtable(
            {"category": ["A", "A"], "value": [None, None]},
            schema={"category": "string", "value": "int64"},
        )

        expr = t.group_by("category").aggregate(
            min_val=t.value.min(), max_val=t.value.max()
        )
        result = con.execute(expr)

        # MIN/MAX of all NULLs should be NULL
        assert result["min_val"].iloc[0] is None or result["min_val"].isna().iloc[0]
        assert result["max_val"].iloc[0] is None or result["max_val"].isna().iloc[0]


@pytest.mark.usefixtures("con")
class TestAggregateInStreamingContext:
    """Test aggregates combined with streaming/temporal features."""

    def test_aggregate_with_mz_now_filter(self, con):
        """Test GROUP BY aggregate combined with mz_now() temporal filter."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
                "created_at": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ],
            }
        )
        t = t.mutate(created_at=t.created_at.cast("timestamp"))

        # Aggregate with temporal filter
        expr = (
            t.filter(mz_now() > t.created_at + ibis.interval(hours=1))
            .group_by("category")
            .aggregate(total=t.value.sum())
        )

        # Should compile successfully
        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "group by" in sql.lower()

    def test_aggregate_with_mz_now_in_select(self, con):
        """Test adding mz_now() to aggregate results."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
            }
        )

        # Add mz_now() as a column in aggregate
        expr = t.group_by("category").aggregate(
            total=t.value.sum(), snapshot_time=mz_now()
        )

        sql = con.compile(expr)
        assert "mz_now()" in sql.lower()
        assert "group by" in sql.lower()

    def test_distinct_aggregate_variations(self, con):
        """Test various DISTINCT aggregate patterns."""
        t = ibis.memtable(
            {
                "category": ["A", "A", "B", "B", "A"],
                "tag": ["x", "y", "x", "z", "x"],
            }
        )

        # COUNT DISTINCT
        expr = t.group_by("category").aggregate(distinct_tags=t.tag.nunique())

        result = con.execute(expr)
        # Category A has tags: x, y, x -> 2 distinct
        # Category B has tags: x, z -> 2 distinct
        assert len(result) == 2
