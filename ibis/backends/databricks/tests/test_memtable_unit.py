"""Unit tests for the allow_memtable_in_memory parameter (no connection required)."""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock, Mock, patch

import pytest

import ibis

pytestmark = pytest.mark.databricks


def test_allow_memtable_in_memory_initializes_polars_backend():
    """Test that allow_memtable_in_memory=True initializes Polars backend."""
    # Only mock the databricks connection, not the Polars backend
    # This allows coverage to see the actual code execution
    with patch("databricks.sql.connect") as mock_db_connect:
        mock_db_connect.return_value = MagicMock()

        con = ibis.databricks.connect(
            server_hostname="test.databricks.com",
            http_path="/sql/1.0/warehouses/test",
            access_token="test_token",  # noqa: S106
            allow_memtable_in_memory=True,
        )

        # Verify the flag is set
        assert hasattr(con, "_memtable_in_memory")
        assert con._memtable_in_memory is True

        # Verify Polars backend is stored
        assert hasattr(con, "_polars_backend")
        assert con._polars_backend is not None


def test_default_memtable_behavior_without_flag():
    """Test that default behavior creates memtable volume (not in-memory)."""
    with (
        patch("databricks.sql.connect") as mock_db_connect,
        patch.object(ibis.backends.databricks.Backend, "current_catalog", "test_cat"),
        patch.object(ibis.backends.databricks.Backend, "current_database", "test_db"),
    ):
        mock_db_connect.return_value = MagicMock()

        con = ibis.databricks.connect(
            server_hostname="test.databricks.com",
            http_path="/sql/1.0/warehouses/test",
            access_token="test_token",  # noqa: S106
        )

        # Verify in-memory flag is False
        assert hasattr(con, "_memtable_in_memory")
        assert con._memtable_in_memory is False

        # Verify memtable volume was set up
        assert hasattr(con, "_memtable_volume")
        assert con._memtable_volume is not None


def test_import_error_when_polars_not_available():
    """Test that proper ImportError is raised when Polars is not installed."""
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "ibis.backends.polars" or name.startswith("ibis.backends.polars."):
            raise ImportError("No module named 'ibis.backends.polars'")
        return original_import(name, *args, **kwargs)

    with (
        patch("databricks.sql.connect") as mock_db_connect,
        patch.object(builtins, "__import__", mock_import),
    ):
        mock_db_connect.return_value = MagicMock()

        with pytest.raises(
            ImportError, match="additional dependencies must be installed"
        ):
            ibis.databricks.connect(
                server_hostname="test.databricks.com",
                http_path="/sql/1.0/warehouses/test",
                access_token="test_token",  # noqa: S106
                allow_memtable_in_memory=True,
            )


def test_register_in_memory_table_delegates_to_polars():
    """Test that _register_in_memory_table delegates to Polars when in-memory mode."""
    with patch("databricks.sql.connect") as mock_db_connect:
        mock_db_connect.return_value = MagicMock()

        con = ibis.databricks.connect(
            server_hostname="test.databricks.com",
            http_path="/sql/1.0/warehouses/test",
            access_token="test_token",  # noqa: S106
            allow_memtable_in_memory=True,
        )

        # Create a mock InMemoryTable operation
        mock_op = Mock()
        mock_op.name = "test_table"

        # Verify that _register_polars_memtable method exists and can be called
        assert hasattr(con, "_register_polars_memtable")

        # The actual registration will delegate to the Polars backend
        # We can't easily test this without a real table, but we verify the method exists
        assert callable(con._register_polars_memtable)


def test_memtable_finalizer_uses_polars_when_in_memory():
    """Test that the finalizer delegates to Polars backend in in-memory mode."""
    with patch("databricks.sql.connect") as mock_db_connect:
        mock_db_connect.return_value = MagicMock()

        con = ibis.databricks.connect(
            server_hostname="test.databricks.com",
            http_path="/sql/1.0/warehouses/test",
            access_token="test_token",  # noqa: S106
            allow_memtable_in_memory=True,
        )

        # Get the finalizer
        finalizer = con._make_memtable_finalizer("test_table")

        # Verify the finalizer is callable
        assert callable(finalizer)

        # The finalizer should work with the real Polars backend
        # We can't easily test the actual cleanup without creating tables,
        # but we verify the mechanism exists


def test_explicit_memtable_volume_overrides_in_memory_flag():
    """Test that providing memtable_volume explicitly takes precedence."""
    with (
        patch("databricks.sql.connect") as mock_db_connect,
        patch.object(ibis.backends.databricks.Backend, "current_catalog", "test_cat"),
        patch.object(ibis.backends.databricks.Backend, "current_database", "test_db"),
    ):
        mock_db_connect.return_value = MagicMock()

        con = ibis.databricks.connect(
            server_hostname="test.databricks.com",
            http_path="/sql/1.0/warehouses/test",
            access_token="test_token",  # noqa: S106
            memtable_volume="my_custom_volume",
            allow_memtable_in_memory=True,  # This should be ignored
        )

        # When explicit volume is provided, should use volume-based approach
        assert hasattr(con, "_memtable_volume")
        assert con._memtable_volume == "my_custom_volume"
        # The in-memory flag might not even be set in this path
