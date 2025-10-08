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

    def test_create_and_drop_counter_source(self, con):
        """Test creating and dropping a COUNTER source."""
        import time

        source_name = util.gen_name("counter_src")

        try:
            # Create counter source
            counter = con.create_source(
                source_name,
                connector="COUNTER",
                properties={"TICK INTERVAL": "100ms"},
            )

            # Verify it exists
            assert source_name in con.list_sources()

            # Wait a bit for some data to generate
            time.sleep(0.5)

            # Query the counter
            result = counter.limit(10).execute()
            assert len(result) > 0
            assert "counter" in result.columns
            # Counter should have sequential values
            assert result["counter"].min() >= 1
        finally:
            con.drop_source(source_name, force=True)

    def test_create_auction_source_with_all_tables(self, con):
        """Test creating AUCTION source with FOR ALL TABLES."""
        import time

        source_name = util.gen_name("auction_src")

        try:
            # Create auction source
            con.create_source(
                source_name,
                connector="AUCTION",
                properties={"TICK INTERVAL": "100ms"},
                for_all_tables=True,
            )

            # Verify it exists
            assert source_name in con.list_sources()

            # Wait for data to generate (auction needs more time than counter)
            time.sleep(3.0)

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
        finally:
            # Clean up - CASCADE will drop dependent subsources
            con.drop_source(source_name, force=True, cascade=True)

    def test_list_sources(self, con):
        """Test listing sources."""
        source_names = [util.gen_name("list_src") for _ in range(2)]

        try:
            # Create multiple sources
            for name in source_names:
                con.create_source(
                    name,
                    connector="COUNTER",
                    properties={"TICK INTERVAL": "1s"},
                )

            # List sources
            all_sources = con.list_sources()

            # Verify our sources are in the list
            for name in source_names:
                assert name in all_sources, f"{name} not found in {all_sources}"
        finally:
            # Cleanup
            for name in source_names:
                con.drop_source(name, force=True)

    def test_list_sources_with_like(self, con):
        """Test listing sources with LIKE pattern."""
        prefix = util.gen_name("like_src_test")
        source_names = [f"{prefix}_{i}" for i in range(2)]
        other_name = util.gen_name("other_src")

        try:
            # Create sources with specific prefix
            for name in source_names:
                con.create_source(
                    name,
                    connector="COUNTER",
                    properties={"TICK INTERVAL": "1s"},
                )

            # Create one with different prefix
            con.create_source(
                other_name,
                connector="COUNTER",
                properties={"TICK INTERVAL": "1s"},
            )

            # List with LIKE pattern
            filtered_sources = con.list_sources(like=f"{prefix}%")

            # Verify only matching sources are returned
            for name in source_names:
                assert name in filtered_sources
            assert other_name not in filtered_sources
        finally:
            # Cleanup
            for name in source_names + [other_name]:
                con.drop_source(name, force=True)

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

    def test_materialized_view_over_load_generator(self, con):
        """Test creating a materialized view over a load generator source."""
        import time

        source_name = util.gen_name("mv_counter_src")
        mv_name = util.gen_name("counter_mv")

        try:
            # Create counter source
            counter = con.create_source(
                source_name,
                connector="COUNTER",
                properties={"TICK INTERVAL": "100ms"},
            )

            # Create MV over the source
            expr = counter.limit(100)
            mv = con.create_materialized_view(mv_name, expr)

            # Wait for some data
            time.sleep(0.5)

            # Query the MV
            result = mv.limit(10).execute()
            assert len(result) > 0
            assert "counter" in result.columns
        finally:
            # Cleanup - drop MV first, then source
            con.drop_materialized_view(mv_name, force=True)
            con.drop_source(source_name, force=True)

    def test_counter_source_generates_sequential_data(self, con):
        """Test that COUNTER generates sequential numbers."""
        import time

        source_name = util.gen_name("seq_counter")

        try:
            counter = con.create_source(
                source_name,
                connector="COUNTER",
                properties={"TICK INTERVAL": "50ms"},
            )

            # Wait for multiple ticks
            time.sleep(0.3)

            # Query data
            result = counter.order_by("counter").limit(10).execute()
            assert len(result) > 0

            # Verify sequential nature
            counters = result["counter"].tolist()
            # Should start from 1
            assert counters[0] >= 1
            # Should be sequential (allowing for some variation in timing)
            if len(counters) > 1:
                # Check that values are increasing
                assert counters[-1] > counters[0]
        finally:
            con.drop_source(source_name, force=True)

    def test_source_appears_in_catalog(self, con):
        """Test that created source appears in mz_sources catalog."""
        source_name = util.gen_name("catalog_src")

        try:
            # Create a source
            con.create_source(
                source_name,
                connector="COUNTER",
                properties={"TICK INTERVAL": "1s"},
            )

            # Query catalog to find it
            result = con.sql(f"""
                SELECT name, type
                FROM mz_catalog.mz_sources
                WHERE name = '{source_name}'
            """).execute()

            assert len(result) == 1
            assert result["name"].iloc[0] == source_name
            assert result["type"].iloc[0] == "load-generator"
        finally:
            con.drop_source(source_name, force=True)


class TestSourceAPI:
    """Tests for the unified create_source API.

    These tests demonstrate the flexible API that supports all Materialize
    source types while maintaining compatibility with RisingWave.
    """

    def test_create_source_with_new_load_generator_api(self, con):
        """Test load generator using RisingWave-compatible API."""
        import time

        source_name = util.gen_name("new_api_counter")

        try:
            # Use RisingWave-style API with connector and properties
            counter = con.create_source(
                source_name,
                connector="COUNTER",
                properties={"TICK INTERVAL": "100ms"},
            )

            time.sleep(0.3)
            result = counter.limit(5).execute()
            assert len(result) > 0
            assert "counter" in result.columns
        finally:
            con.drop_source(source_name, force=True)

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
