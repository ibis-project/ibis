"""Tests for Materialize connections.

Tests cover connection creation, management, and operations.
"""

from __future__ import annotations


class TestConnections:
    """Tests for connection operations (create, drop, list).

    Note: These tests use list operations only, as create/drop would require
    actual external systems (Kafka brokers, PostgreSQL instances, etc.).
    """

    def test_list_connections(self, con):
        """Test listing connections."""
        # Should not error even if no connections exist
        connections = con.list_connections()
        assert isinstance(connections, list)

    def test_list_connections_with_like(self, con):
        """Test listing connections with LIKE pattern."""
        # This should work even if no matches
        connections = con.list_connections(like="nonexistent%")
        assert isinstance(connections, list)
        assert len(connections) == 0
