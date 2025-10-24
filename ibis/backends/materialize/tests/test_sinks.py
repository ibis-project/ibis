"""Tests for Materialize sinks.

Tests cover sink creation, management, and operations.
"""

from __future__ import annotations


class TestSinks:
    """Tests for sink operations (create, drop, list).

    Note: These tests use mock connections and don't actually send data to Kafka.
    Real Kafka integration would require external infrastructure.
    """

    def test_list_sinks(self, con):
        """Test listing sinks."""
        # Should not error even if no sinks exist
        sinks = con.list_sinks()
        assert isinstance(sinks, list)

    def test_list_sinks_with_like(self, con):
        """Test listing sinks with LIKE pattern."""
        # This should work even if no matches
        sinks = con.list_sinks(like="nonexistent%")
        assert isinstance(sinks, list)
        assert len(sinks) == 0


class TestSinkAPI:
    """Tests documenting the sink API.

    These tests demonstrate the API patterns without executing real sink creation,
    since that would require Kafka infrastructure.
    """

    def test_sink_from_materialized_view_documented(self):
        """Document creating sink from materialized view.

        In real usage, user would first create connection and materialized view:
        >>> con.raw_sql("CREATE CONNECTION kafka_conn TO KAFKA ...")
        >>> mv = con.create_materialized_view("orders_mv", orders_expr)

        Then create a sink:
        >>> con.create_sink(
        ...     "orders_sink",
        ...     sink_from="orders_mv",
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "orders"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     envelope="UPSERT",
        ...     key=["order_id"],
        ... )
        """

    def test_sink_from_expression_documented(self):
        """Document creating sink from expression (RisingWave style).

        RisingWave-compatible pattern using obj parameter:
        >>> orders = con.table("orders")
        >>> expr = orders.filter(orders.status == "complete")
        >>> con.create_sink(
        ...     "completed_orders",
        ...     obj=expr,
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "completed_orders"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     envelope="UPSERT",
        ...     key=["order_id"],
        ... )
        """

    def test_sink_with_key_value_formats_documented(self):
        """Document sink with separate key and value formats.

        >>> con.create_sink(
        ...     "events_sink",
        ...     sink_from="events_mv",
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "events"},
        ...     format_spec={"KEY FORMAT": "TEXT", "VALUE FORMAT": "JSON"},
        ...     envelope="UPSERT",
        ...     key=["event_id"],
        ... )
        """

    def test_sink_with_debezium_envelope_documented(self):
        """Document sink with DEBEZIUM envelope.

        DEBEZIUM envelope captures before/after state changes:
        >>> con.create_sink(
        ...     "changes_sink",
        ...     sink_from="user_changes_mv",
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "user_changes"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     envelope="DEBEZIUM",
        ... )
        """

    def test_drop_sink_documented(self):
        """Document dropping sinks.

        >>> con.drop_sink("my_sink", force=True)
        """

    def test_list_sinks_documented(self):
        """Document listing sinks.

        >>> con.list_sinks()
        ['orders_sink', 'events_sink']

        >>> con.list_sinks(like="orders%")
        ['orders_sink']
        """
