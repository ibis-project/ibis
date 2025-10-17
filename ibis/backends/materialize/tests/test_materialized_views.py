"""Tests for Materialize materialized views.

Tests cover materialized view operations and catalog queries.
"""

from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis
from ibis import util


class TestMaterializedViews:
    """Tests for materialized view operations."""

    def test_create_and_drop_materialized_view(self, con, alltypes):
        """Test creating and dropping a materialized view."""
        mv_name = util.gen_name("test_mv")
        expr = alltypes[["string_col", "double_col"]].distinct()

        # Create MV
        mv = con.create_materialized_view(mv_name, expr)

        # Verify it exists
        assert mv_name in con.list_materialized_views()

        # Query it
        result = mv.limit(5).execute()
        assert len(result) <= 5
        assert "string_col" in result.columns
        assert "double_col" in result.columns

        # Drop it
        con.drop_materialized_view(mv_name)
        assert mv_name not in con.list_materialized_views()

    def test_materialized_view_with_aggregation(self, con, alltypes):
        """Test MV with GROUP BY aggregation."""
        mv_name = util.gen_name("agg_mv")

        try:
            expr = alltypes.group_by("string_col").aggregate(
                count=alltypes.count(), avg_double=alltypes.double_col.mean()
            )

            mv = con.create_materialized_view(mv_name, expr)
            result = mv.execute()

            # Verify aggregation worked
            assert "string_col" in result.columns
            assert "count" in result.columns
            assert "avg_double" in result.columns
            assert len(result) > 0

            # Verify results match direct query
            expected = expr.execute()
            tm.assert_frame_equal(
                result.sort_values("string_col").reset_index(drop=True),
                expected.sort_values("string_col").reset_index(drop=True),
            )
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_materialized_view_with_join(self, con):
        """Test MV with JOIN between tables."""
        mv_name = util.gen_name("join_mv")

        try:
            batting = con.table("batting")
            awards = con.table("awards_players")

            # Create MV with a join
            expr = (
                batting.join(awards, batting.playerID == awards.playerID)
                .select(
                    batting.playerID,
                    batting.yearID,
                    batting.teamID,
                    awards.awardID,
                )
                .limit(100)
            )

            mv = con.create_materialized_view(mv_name, expr)
            result = mv.limit(10).execute()

            # Verify join worked
            assert "playerID" in result.columns
            assert "teamID" in result.columns
            assert "awardID" in result.columns
            assert len(result) > 0
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_materialized_view_overwrite(self, con, alltypes):
        """Test overwriting an existing MV."""
        mv_name = util.gen_name("overwrite_mv")

        try:
            # Create first MV
            expr1 = alltypes[["int_col"]].limit(10)
            mv1 = con.create_materialized_view(mv_name, expr1)
            result1 = mv1.limit(1).execute()
            assert list(result1.columns) == ["int_col"]

            # Overwrite with different query
            expr2 = alltypes[["string_col"]].limit(10)
            mv2 = con.create_materialized_view(mv_name, expr2, overwrite=True)
            result2 = mv2.limit(1).execute()
            assert list(result2.columns) == ["string_col"]
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_list_materialized_views(self, con, alltypes):
        """Test listing materialized views."""
        mv_names = [util.gen_name("list_mv") for _ in range(3)]

        try:
            # Create multiple MVs
            for name in mv_names:
                expr = alltypes[["id"]].limit(10)
                con.create_materialized_view(name, expr)

            # List MVs
            all_mvs = con.list_materialized_views()

            # Verify our MVs are in the list
            for name in mv_names:
                assert name in all_mvs, f"{name} not found in {all_mvs}"
        finally:
            # Cleanup
            for name in mv_names:
                con.drop_materialized_view(name, force=True)

    def test_list_materialized_views_with_like(self, con, alltypes):
        """Test listing materialized views with LIKE pattern."""
        prefix = util.gen_name("like_test")
        mv_names = [f"{prefix}_mv_{i}" for i in range(3)]
        other_name = util.gen_name("other_mv")

        try:
            # Create MVs with specific prefix
            for name in mv_names:
                expr = alltypes[["id"]].limit(10)
                con.create_materialized_view(name, expr)

            # Create one with different prefix
            expr = alltypes[["id"]].limit(10)
            con.create_materialized_view(other_name, expr)

            # List with LIKE pattern
            filtered_mvs = con.list_materialized_views(like=f"{prefix}%")

            # Verify only matching MVs are returned
            for name in mv_names:
                assert name in filtered_mvs
            assert other_name not in filtered_mvs
        finally:
            # Cleanup
            for name in mv_names + [other_name]:
                con.drop_materialized_view(name, force=True)

    def test_materialized_view_with_filter(self, con, alltypes):
        """Test MV with WHERE clause."""
        mv_name = util.gen_name("filter_mv")

        try:
            expr = alltypes.filter(alltypes.int_col > 0)[["int_col", "double_col"]]
            mv = con.create_materialized_view(mv_name, expr)
            result = mv.execute()

            # Verify filter worked
            assert len(result) > 0
            assert all(result["int_col"] > 0)
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_materialized_view_with_order_by(self, con, alltypes):
        """Test MV with ORDER BY clause."""
        mv_name = util.gen_name("order_mv")

        try:
            expr = (
                alltypes[["int_col", "string_col"]]
                .order_by(alltypes.int_col.desc())
                .limit(20)
            )
            mv = con.create_materialized_view(mv_name, expr)
            result = mv.execute()

            # Verify MV was created and returns results
            assert len(result) > 0
            assert "int_col" in result.columns
            assert "string_col" in result.columns
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_drop_nonexistent_materialized_view_with_force(self, con):
        """Test dropping non-existent MV with force=True doesn't error."""
        mv_name = util.gen_name("nonexistent_mv")

        # Should not raise an error
        con.drop_materialized_view(mv_name, force=True)

    def test_drop_nonexistent_materialized_view_without_force(self, con):
        """Test dropping non-existent MV without force raises error."""
        mv_name = util.gen_name("nonexistent_mv")

        # Should raise an error (psycopg will raise an exception)
        with pytest.raises(Exception):  # noqa: B017
            con.drop_materialized_view(mv_name, force=False)


class TestMaterializeCatalog:
    """Tests for Materialize-specific catalog queries.

    These tests query mz_catalog system tables to verify metadata
    about streaming objects like sources, sinks, and materialized views.
    """

    def test_query_mz_materialized_views_catalog(self, con):
        """Test querying mz_materialized_views catalog directly."""
        result = con.sql("""
            SELECT name, id
            FROM mz_catalog.mz_materialized_views
            LIMIT 5
        """).execute()

        assert "name" in result.columns
        assert "id" in result.columns
        # Should have system materialized views even if we haven't created any
        assert len(result) >= 0

    def test_query_mz_sources_catalog(self, con):
        """Test querying mz_sources catalog."""
        result = con.sql("""
            SELECT name, type
            FROM mz_catalog.mz_sources
            LIMIT 10
        """).execute()

        assert "name" in result.columns
        assert "type" in result.columns

    def test_query_mz_tables_catalog(self, con):
        """Test querying mz_tables catalog."""
        result = con.sql("""
            SELECT name, id
            FROM mz_catalog.mz_tables
            WHERE name IN ('functional_alltypes', 'batting', 'awards_players')
        """).execute()

        assert "name" in result.columns
        assert "id" in result.columns
        # Should find our test tables
        assert len(result) >= 1

    def test_query_mz_views_catalog(self, con):
        """Test querying mz_views catalog."""
        result = con.sql("""
            SELECT name, id
            FROM mz_catalog.mz_views
            LIMIT 10
        """).execute()

        assert "name" in result.columns
        assert "id" in result.columns

    def test_query_mz_columns_catalog(self, con):
        """Test querying mz_columns catalog for table structure."""
        result = con.sql("""
            SELECT c.name AS column_name, c.type AS data_type
            FROM mz_catalog.mz_columns c
            JOIN mz_catalog.mz_tables t ON c.id = t.id
            WHERE t.name = 'functional_alltypes'
            ORDER BY c.position
            LIMIT 10
        """).execute()

        assert "column_name" in result.columns
        assert "data_type" in result.columns
        # functional_alltypes has many columns
        assert len(result) > 0

    def test_materialized_view_appears_in_catalog(self, con, alltypes):
        """Test that created MV appears in mz_materialized_views catalog."""
        mv_name = util.gen_name("catalog_test_mv")

        try:
            # Create a materialized view
            expr = alltypes[["id", "string_col"]].limit(10)
            con.create_materialized_view(mv_name, expr)

            # Query catalog to find it
            result = con.sql(f"""
                SELECT name, id
                FROM mz_catalog.mz_materialized_views
                WHERE name = '{mv_name}'
            """).execute()

            assert len(result) == 1
            assert result["name"].iloc[0] == mv_name
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_query_mz_schemas_catalog(self, con):
        """Test querying mz_schemas catalog."""
        result = con.sql("""
            SELECT name, id
            FROM mz_catalog.mz_schemas
            WHERE name IN ('public', 'mz_catalog', 'mz_temp', 'information_schema')
            ORDER BY name
        """).execute()

        assert "name" in result.columns
        assert "id" in result.columns
        # Should have standard schemas
        assert len(result) >= 2

    def test_query_mz_databases_catalog(self, con):
        """Test querying mz_databases catalog."""
        result = con.sql("""
            SELECT name, id
            FROM mz_catalog.mz_databases
            ORDER BY name
        """).execute()

        assert "name" in result.columns
        assert "id" in result.columns
        # Should have at least materialize database
        assert len(result) >= 1


class TestMaterializeStreaming:
    """Tests for Materialize streaming-specific behaviors.

    These tests verify that materialized views behave correctly
    for streaming/incremental computation use cases.
    """

    def test_materialized_view_distinct_behavior(self, con, alltypes):
        """Test that MV maintains distinct values correctly."""
        mv_name = util.gen_name("distinct_mv")

        try:
            # Create MV with distinct values
            expr = alltypes[["string_col"]].distinct()
            mv = con.create_materialized_view(mv_name, expr)

            # Query the MV
            result = mv.execute()

            # Verify distinct constraint
            assert len(result) == len(result["string_col"].unique())

            # Compare with direct query
            expected = expr.execute()
            assert len(result) == len(expected)
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_materialized_view_with_window_function(self, con, alltypes):
        """Test MV with window function."""
        mv_name = util.gen_name("window_mv")

        try:
            expr = alltypes.mutate(
                row_num=ibis.row_number().over(ibis.window(order_by=alltypes.id))
            )[["id", "string_col", "row_num"]].limit(20)

            mv = con.create_materialized_view(mv_name, expr)
            result = mv.execute()

            # Verify window function worked
            assert "row_num" in result.columns
            assert len(result) > 0
        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_materialized_view_with_self_join(self, con, alltypes):
        """Test MV with self-join."""
        mv_name = util.gen_name("self_join_mv")

        try:
            t1 = alltypes.select("id", "int_col")
            t2 = alltypes.select("id", "double_col")

            expr = t1.join(t2, "id").limit(50)

            mv = con.create_materialized_view(mv_name, expr)
            result = mv.execute()

            # Verify self-join worked
            assert "int_col" in result.columns
            assert "double_col" in result.columns
            assert len(result) > 0
        finally:
            con.drop_materialized_view(mv_name, force=True)
