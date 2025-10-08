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


class TestConnectionAPI:
    """Tests documenting the connection API.

    These tests demonstrate the API patterns without executing real connection creation,
    since that would require external infrastructure (Kafka brokers, databases, etc.).
    """

    def test_kafka_connection_documented(self):
        """Document creating Kafka connection with SASL authentication.

        >>> con.create_secret("kafka_password", "my_secret_password")
        >>> con.create_connection(
        ...     "kafka_conn",
        ...     connection_type="KAFKA",
        ...     properties={
        ...         "BROKER": "localhost:9092",
        ...         "SASL MECHANISMS": "PLAIN",
        ...         "SASL USERNAME": "user",
        ...         "SASL PASSWORD": SECRET("kafka_password"),
        ...     },
        ... )
        """

    def test_postgres_connection_documented(self):
        """Document creating PostgreSQL CDC connection.

        >>> con.create_secret("pg_password", "postgres_pwd")
        >>> con.create_connection(
        ...     "pg_conn",
        ...     connection_type="POSTGRES",
        ...     properties={
        ...         "HOST": "localhost",
        ...         "PORT": "5432",
        ...         "DATABASE": "mydb",
        ...         "USER": "postgres",
        ...         "PASSWORD": SECRET("pg_password"),
        ...     },
        ... )
        """

    def test_mysql_connection_documented(self):
        """Document creating MySQL CDC connection.

        >>> con.create_secret("mysql_password", "mysql_pwd")
        >>> con.create_connection(
        ...     "mysql_conn",
        ...     connection_type="MYSQL",
        ...     properties={
        ...         "HOST": "localhost",
        ...         "PORT": "3306",
        ...         "USER": "mysql",
        ...         "PASSWORD": SECRET("mysql_password"),
        ...     },
        ... )
        """

    def test_aws_connection_documented(self):
        """Document creating AWS connection for S3 sources.

        >>> con.create_secret("aws_key", "AKIAIOSFODNN7EXAMPLE")
        >>> con.create_secret("aws_secret", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        >>> con.create_connection(
        ...     "aws_conn",
        ...     connection_type="AWS",
        ...     properties={
        ...         "REGION": "us-east-1",
        ...         "ACCESS KEY ID": SECRET("aws_key"),
        ...         "SECRET ACCESS KEY": SECRET("aws_secret"),
        ...     },
        ... )
        """

    def test_ssh_tunnel_connection_documented(self):
        """Document creating SSH tunnel connection.

        >>> con.create_connection(
        ...     "ssh_conn",
        ...     connection_type="SSH TUNNEL",
        ...     properties={"HOST": "bastion.example.com", "PORT": "22", "USER": "tunnel_user"},
        ... )
        """

    def test_connection_without_validation_documented(self):
        """Document creating connection without validation.

        Useful for testing or when the external system is temporarily unavailable:
        >>> con.create_connection(
        ...     "kafka_conn",
        ...     connection_type="KAFKA",
        ...     properties={"BROKER": "localhost:9092"},
        ...     validate=False,
        ... )
        """

    def test_drop_connection_documented(self):
        """Document dropping connections.

        >>> con.drop_connection("kafka_conn", force=True)
        >>> con.drop_connection("pg_conn", cascade=True)
        """

    def test_list_connections_documented(self):
        """Document listing connections.

        >>> con.list_connections()
        ['kafka_conn', 'pg_conn', 'aws_conn']

        >>> con.list_connections(like="kafka%")
        ['kafka_conn']
        """
