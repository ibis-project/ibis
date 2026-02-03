"""Tests for Materialize sources and load generators.

Tests cover source creation, management, and load generator functionality.
"""

from __future__ import annotations

import pytest

from ibis import util


class TestLoadGenerators:
    """Tests for Materialize load generator sources.

    Load generators create synthetic data for testing and demonstrations.
    """

    def test_create_and_drop_auction_source(self, con, auction_source):
        """Test AUCTION source exists and can be queried."""
        # Verify source exists
        assert auction_source in con.list_sources()

        # Query the bids subsource
        bids = con.table("bids")
        result = bids.limit(10).execute()
        assert len(result) > 0
        assert "id" in result.columns

    def test_auction_source_subsources(self, con, auction_source):
        """Test AUCTION source creates all expected subsources."""
        _ = auction_source  # Fixture needed to create auction source
        # Query the bids subsource
        bids = con.table("bids")
        result = bids.limit(5).execute()

        # Verify auction data structure
        assert len(result) > 0
        assert "id" in result.columns
        assert "buyer" in result.columns
        assert "auction_id" in result.columns
        assert "amount" in result.columns

        # Query the auctions subsource
        auctions = con.table("auctions")
        auction_result = auctions.limit(5).execute()
        assert len(auction_result) > 0
        assert "id" in auction_result.columns

    def test_list_sources(self, con, auction_source):
        """Test listing sources includes our auction source."""
        # List sources
        all_sources = con.list_sources()

        # Verify our auction source is in the list
        assert auction_source in all_sources, (
            f"{auction_source} not found in {all_sources}"
        )

    def test_list_sources_with_like(self, con, auction_source):
        """Test listing sources with LIKE pattern."""
        # List with LIKE pattern matching our source
        filtered_sources = con.list_sources(like="test_auction%")

        # Verify our auction source is matched
        assert auction_source in filtered_sources

    def test_drop_nonexistent_source_with_force(self, con):
        """Test dropping non-existent source with force=True doesn't error."""
        source_name = util.gen_name("nonexistent_src")

        # Should not raise an error
        con.drop_source(source_name, force=True)

    def test_drop_nonexistent_source_without_force(self, con):
        """Test dropping non-existent source without force raises error."""
        source_name = util.gen_name("nonexistent_src")

        # Should raise an error
        with pytest.raises(Exception):  # noqa: B017
            con.drop_source(source_name, force=False)

    def test_materialized_view_over_load_generator(self, con, auction_source):
        """Test creating a materialized view over a load generator source."""
        _ = auction_source  # Fixture needed to create auction source
        mv_name = util.gen_name("auction_mv")
        mv_created = False

        try:
            # Create MV over the bids subsource
            bids = con.table("bids")
            expr = bids.limit(100)
            mv = con.create_materialized_view(mv_name, expr)
            mv_created = True

            # Query the MV
            result = mv.limit(10).execute()
            assert len(result) > 0
            assert "id" in result.columns
        finally:
            # Cleanup
            if mv_created:
                con.drop_materialized_view(mv_name, force=True)

    def test_source_appears_in_catalog(self, con, auction_source):
        """Test that created source appears in mz_sources catalog."""
        # Query catalog to find it
        result = con.sql(f"""
            SELECT name, type
            FROM mz_catalog.mz_sources
            WHERE name = '{auction_source}'
        """).execute()

        assert len(result) == 1
        assert result["name"].iloc[0] == auction_source
        assert result["type"].iloc[0] == "load-generator"


class TestSourceAPI:
    """Tests for the unified create_source API.

    These tests demonstrate the flexible API that supports all Materialize
    source types while maintaining compatibility with RisingWave.
    """

    def test_create_source_with_new_load_generator_api(self, con, auction_source):
        """Test load generator creates queryable subsources."""
        _ = auction_source  # Fixture needed to create auction source
        # Verify we can query the bids subsource
        bids = con.table("bids")
        result = bids.limit(5).execute()
        assert len(result) > 0
        assert "id" in result.columns

    def test_kafka_source_api_documented(self):
        """Document Kafka source API.

        In real usage, user would first create a connection:
        >>> con.raw_sql("CREATE CONNECTION kafka_conn TO KAFKA ...")

        Then create a Kafka source:
        >>> kafka_src = con.create_source(
        ...     "kafka_data",
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "my_topic"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     envelope="UPSERT",
        ... )
        """

    def test_postgres_source_api_documented(self):
        """Document PostgreSQL CDC source API.

        In real usage, user would first create a connection:
        >>> con.raw_sql("CREATE SECRET pgpass AS 'password'")
        >>> con.raw_sql("CREATE CONNECTION pg_conn TO POSTGRES ...")

        Then create a PostgreSQL source:
        >>> pg_src = con.create_source(
        ...     "pg_tables",
        ...     connector="POSTGRES",
        ...     connection="pg_conn",
        ...     properties={"PUBLICATION": "mz_source"},
        ...     for_all_tables=True,
        ... )
        """

    def test_kafka_include_properties_documented(self):
        """Document Kafka source with INCLUDE properties.

        INCLUDE properties add metadata columns to Kafka sources:
        >>> kafka_src = con.create_source(
        ...     "kafka_with_metadata",
        ...     connector="KAFKA",
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "events"},
        ...     format_spec={"FORMAT": "JSON"},
        ...     include_properties=["KEY", "PARTITION", "OFFSET", "TIMESTAMP"],
        ... )
        """

    def test_kafka_explicit_schema_documented(self):
        """Document Kafka source with explicit schema.

        Some Kafka sources require explicit schema definition:
        >>> schema = ibis.schema(
        ...     [("event_id", "int64"), ("event_name", "string"), ("timestamp", "timestamp")]
        ... )
        >>> kafka_src = con.create_source(
        ...     "kafka_events",
        ...     connector="KAFKA",
        ...     schema=schema,
        ...     connection="kafka_conn",
        ...     properties={"TOPIC": "events"},
        ...     format_spec={"FORMAT": "JSON"},
        ... )
        """

    def test_postgres_for_tables_documented(self):
        """Document PostgreSQL source with specific tables.

        Select specific tables from a PostgreSQL database:
        >>> pg_src = con.create_source(
        ...     "pg_specific_tables",
        ...     connector="POSTGRES",
        ...     connection="pg_conn",
        ...     properties={"PUBLICATION": "mz_source"},
        ...     for_tables=[("public.users", "users"), ("public.orders", "orders")],
        ... )
        """

    def test_postgres_for_schemas_documented(self):
        """Document PostgreSQL source with specific schemas.

        Select specific schemas from a PostgreSQL database:
        >>> pg_src = con.create_source(
        ...     "pg_schemas",
        ...     connector="POSTGRES",
        ...     connection="pg_conn",
        ...     properties={"PUBLICATION": "mz_source"},
        ...     for_schemas=["public", "analytics"],
        ... )
        """
