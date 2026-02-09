"""Tests for Materialize indexes.

Tests cover index creation, management, and operations.
"""

from __future__ import annotations

from ibis.util import gen_name


def test_list_indexes(con):
    """Test listing indexes."""
    indexes = con.list_indexes()
    assert isinstance(indexes, list)


def test_list_indexes_on_table(con):
    """Test listing indexes for a specific table."""
    # functional_alltypes may have default indexes
    indexes = con.list_indexes(table="functional_alltypes")
    assert isinstance(indexes, list)


def test_list_indexes_with_like(con):
    """Test listing indexes with LIKE pattern."""
    indexes = con.list_indexes(like="nonexistent%")
    assert isinstance(indexes, list)
    assert len(indexes) == 0


def test_create_and_drop_index(con):
    """Test creating and dropping an index."""
    mv_name = gen_name("test_mv")
    idx_name = gen_name("test_idx")

    # Create a materialized view to index
    expr = con.table("functional_alltypes").select("id", "int_col")
    con.create_materialized_view(mv_name, expr)

    # Create index
    con.create_index(idx_name, mv_name, expressions=["int_col"])

    try:
        # Verify it exists
        indexes = con.list_indexes(table=mv_name)
        assert idx_name in indexes

        # Drop index
        con.drop_index(idx_name)

        # Verify it's gone
        indexes = con.list_indexes(table=mv_name)
        assert idx_name not in indexes
    finally:
        # Cleanup
        con.drop_index(idx_name, force=True)
        con.drop_materialized_view(mv_name, force=True)
