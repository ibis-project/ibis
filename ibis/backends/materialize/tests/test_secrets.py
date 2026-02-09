"""Tests for Materialize secrets.

Tests cover secret creation, management, and operations.
"""

from __future__ import annotations


class TestSecrets:
    """Tests for secret operations (create, drop, list).

    Secrets store sensitive data for use in connections.
    """

    def test_list_secrets(self, con):
        """Test listing secrets."""
        # Should not error even if no secrets exist
        secrets = con.list_secrets()
        assert isinstance(secrets, list)

    def test_list_secrets_with_like(self, con):
        """Test listing secrets with LIKE pattern."""
        # This should work even if no matches
        secrets = con.list_secrets(like="nonexistent%")
        assert isinstance(secrets, list)
        assert len(secrets) == 0


class TestSecretAPI:
    """Tests documenting the secret API.

    These tests demonstrate the API patterns without executing real secret creation.
    """

    def test_create_secret_documented(self):
        """Document creating secrets.

        >>> con.create_secret("kafka_password", "my_secret_password")
        >>> con.create_secret("pg_password", "postgres_pwd")
        >>> con.create_secret("api_key", "sk-1234567890abcdef")
        """

    def test_drop_secret_documented(self):
        """Document dropping secrets.

        >>> con.drop_secret("kafka_password", force=True)
        """

    def test_list_secrets_documented(self):
        """Document listing secrets.

        >>> con.list_secrets()
        ['kafka_password', 'pg_password', 'api_key']

        >>> con.list_secrets(like="kafka%")
        ['kafka_password']
        """
