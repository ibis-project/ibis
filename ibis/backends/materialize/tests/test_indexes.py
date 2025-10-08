"""Tests for Materialize indexes.

Tests cover index creation, management, and operations.
"""

from __future__ import annotations


class TestIndexes:
    """Functional tests for index operations."""

    def test_list_indexes(self, con):
        """Test listing indexes."""
        indexes = con.list_indexes()
        assert isinstance(indexes, list)

    def test_list_indexes_on_table(self, con, alltypes):  # noqa: ARG002
        """Test listing indexes for a specific table."""
        # functional_alltypes may have default indexes
        indexes = con.list_indexes(table="functional_alltypes")
        assert isinstance(indexes, list)

    def test_list_indexes_with_like(self, con):
        """Test listing indexes with LIKE pattern."""
        indexes = con.list_indexes(like="nonexistent%")
        assert isinstance(indexes, list)
        assert len(indexes) == 0

    def test_create_and_drop_index(self, con):
        """Test creating and dropping an index."""
        from ibis.util import gen_name

        mv_name = gen_name("test_mv")
        idx_name = gen_name("test_idx")

        try:
            # Create a materialized view to index
            expr = con.table("functional_alltypes").select("id", "int_col")
            con.create_materialized_view(mv_name, expr)

            # Create index
            con.create_index(idx_name, mv_name, expressions=["int_col"])

            # Verify it exists
            indexes = con.list_indexes(table=mv_name)
            assert idx_name in indexes

            # Drop index
            con.drop_index(idx_name)

            # Verify it's gone
            indexes = con.list_indexes(table=mv_name)
            assert idx_name not in indexes
        finally:
            con.drop_index(idx_name, force=True)
            con.drop_materialized_view(mv_name, force=True)


class TestIndexAPI:
    """Documentation tests for index API examples."""

    def test_default_index_documented(self, con):
        """Document creating default index.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> mv_name = "test_default_idx_mv"
        >>> idx_name = "test_default_idx"
        >>> try:
        ...     expr = con.table("functional_alltypes").select("id", "int_col")
        ...     con.create_materialized_view(mv_name, expr)
        ...     con.create_index(idx_name, mv_name)
        ...     indexes = con.list_indexes(table=mv_name)
        ...     assert idx_name in indexes
        ... finally:
        ...     con.drop_index(idx_name, force=True)
        ...     con.drop_materialized_view(mv_name, force=True)
        """

    def test_single_column_index_documented(self, con):
        """Document creating index on single column.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> mv_name = "test_single_col_idx_mv"
        >>> idx_name = "test_single_col_idx"
        >>> try:
        ...     expr = con.table("functional_alltypes").select("id", "int_col")
        ...     con.create_materialized_view(mv_name, expr)
        ...     con.create_index(idx_name, mv_name, expressions=["int_col"])
        ...     indexes = con.list_indexes(table=mv_name)
        ...     assert idx_name in indexes
        ... finally:
        ...     con.drop_index(idx_name, force=True)
        ...     con.drop_materialized_view(mv_name, force=True)
        """

    def test_multi_column_index_documented(self, con):
        """Document creating index on multiple columns.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> mv_name = "test_multi_col_idx_mv"
        >>> idx_name = "test_multi_col_idx"
        >>> try:
        ...     expr = con.table("functional_alltypes").select("id", "int_col", "double_col")
        ...     con.create_materialized_view(mv_name, expr)
        ...     con.create_index(idx_name, mv_name, expressions=["int_col", "double_col"])
        ...     indexes = con.list_indexes(table=mv_name)
        ...     assert idx_name in indexes
        ... finally:
        ...     con.drop_index(idx_name, force=True)
        ...     con.drop_materialized_view(mv_name, force=True)
        """

    def test_expression_index_documented(self, con):
        """Document creating index on expression.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> mv_name = "test_expr_idx_mv"
        >>> idx_name = "test_expr_idx"
        >>> try:
        ...     expr = con.table("functional_alltypes").select("id", "string_col")
        ...     con.create_materialized_view(mv_name, expr)
        ...     con.create_index(idx_name, mv_name, expressions=["upper(string_col)"])
        ...     indexes = con.list_indexes(table=mv_name)
        ...     assert idx_name in indexes
        ... finally:
        ...     con.drop_index(idx_name, force=True)
        ...     con.drop_materialized_view(mv_name, force=True)
        """

    def test_index_in_cluster_documented(self, con):
        """Document creating index in specific cluster.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> mv_name = "test_cluster_idx_mv"
        >>> idx_name = "test_cluster_idx"
        >>> try:
        ...     expr = con.table("functional_alltypes").select("id", "int_col")
        ...     con.create_materialized_view(mv_name, expr)
        ...     # Use quickstart cluster (always available in Materialize)
        ...     con.create_index(idx_name, mv_name, cluster="quickstart")
        ...     indexes = con.list_indexes(table=mv_name)
        ...     assert idx_name in indexes
        ... finally:
        ...     con.drop_index(idx_name, force=True)
        ...     con.drop_materialized_view(mv_name, force=True)
        """

    def test_drop_index_documented(self, con):
        """Document dropping indexes.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> mv_name = "test_drop_idx_mv"
        >>> idx_name = "test_drop_idx"
        >>> try:
        ...     expr = con.table("functional_alltypes").select("id", "int_col")
        ...     con.create_materialized_view(mv_name, expr)
        ...     con.create_index(idx_name, mv_name)
        ...     con.drop_index(idx_name)
        ...     indexes = con.list_indexes(table=mv_name)
        ...     assert idx_name not in indexes
        ... finally:
        ...     con.drop_index(idx_name, force=True)
        ...     con.drop_materialized_view(mv_name, force=True)
        """

    def test_list_indexes_documented(self, con):
        """Document listing indexes.

        >>> import ibis
        >>> con = ibis.materialize.connect()
        >>> mv_name = "test_list_idx_mv"
        >>> idx1_name = "test_list_idx_1"
        >>> idx2_name = "test_list_idx_2"
        >>> try:
        ...     expr = con.table("functional_alltypes").select("id", "int_col")
        ...     con.create_materialized_view(mv_name, expr)
        ...     con.create_index(idx1_name, mv_name, expressions=["id"])
        ...     con.create_index(idx2_name, mv_name, expressions=["int_col"])
        ...
        ...     # List all indexes on the table
        ...     indexes = con.list_indexes(table=mv_name)
        ...     assert idx1_name in indexes
        ...     assert idx2_name in indexes
        ...
        ...     # List with pattern
        ...     indexes_pattern = con.list_indexes(like=f"{idx1_name[:-1]}%")
        ...     assert idx1_name in indexes_pattern
        ... finally:
        ...     con.drop_index(idx1_name, force=True)
        ...     con.drop_index(idx2_name, force=True)
        ...     con.drop_materialized_view(mv_name, force=True)
        """
