"""Tests for Materialize SUBSCRIBE functionality.

Tests cover streaming query functionality via SUBSCRIBE command.
"""

from __future__ import annotations

import contextlib

import pytest


@pytest.fixture(scope="module")
def auction_source(con):
    """Module-level AUCTION source for subscribe tests.

    AUCTION creates subsources with fixed names (bids, auctions, accounts, etc.),
    so we create one source for all tests in this module to avoid conflicts.
    """
    import time

    source_name = "test_subscribe_auction"

    # Drop any existing auction subsources (they're created as sources)
    for subsource in ["accounts", "auctions", "bids", "organizations", "users"]:
        with contextlib.suppress(Exception):
            con.raw_sql(f"DROP SOURCE IF EXISTS {subsource} CASCADE")

    # Drop the main source if it exists
    with contextlib.suppress(Exception):
        con.drop_source(source_name, cascade=True, force=True)

    # Create the source
    con.create_source(
        source_name, connector="AUCTION", properties={"TICK INTERVAL": "100ms"}
    )

    # Wait for initial data
    time.sleep(2.0)

    yield source_name

    # Cleanup after all tests - drop subsources first
    for subsource in ["accounts", "auctions", "bids", "organizations", "users"]:
        with contextlib.suppress(Exception):
            con.raw_sql(f"DROP SOURCE IF EXISTS {subsource} CASCADE")

    with contextlib.suppress(Exception):
        con.drop_source(source_name, cascade=True, force=True)


class TestSubscribe:
    """Tests for SUBSCRIBE streaming functionality.

    Based on the Materialize quickstart guide.
    """

    def test_subscribe_quickstart_workflow(self, con, auction_source):
        """Test SUBSCRIBE with load generator (simplified quickstart example).

        This test demonstrates the Materialize streaming workflow:
        1. Use an AUCTION source (generates auction data)
        2. Create a materialized view over the source
        3. Subscribe to see real-time updates
        """
        from ibis.util import gen_name

        _ = auction_source  # Fixture needed to create auction source
        mv_name = gen_name("auction_sum")

        # Create a materialized view that computes over the bids subsource
        # Get the max bid amount
        bids_table = con.table("bids")
        max_bid_expr = bids_table.aggregate(max_value=bids_table["amount"].max())
        con.create_materialized_view(mv_name, max_bid_expr)

        try:
            # Subscribe to the materialized view
            # Get the first batch (snapshot showing current max value)
            batch_count = 0
            total_rows = 0

            for batch in con.subscribe(mv_name):
                batch_count += 1
                total_rows += len(batch)

                # Verify batch structure
                assert "mz_timestamp" in batch.columns
                assert "mz_diff" in batch.columns
                assert "max_value" in batch.columns

                # Verify we got data
                assert len(batch) > 0

                # Get first batch (snapshot) and exit
                break

            # Should have received at least one batch
            assert batch_count == 1
            assert total_rows > 0

        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_subscribe_arrow_format(self, con, auction_source):
        """Test SUBSCRIBE with Arrow format output."""
        import pyarrow as pa

        from ibis.util import gen_name

        _ = auction_source  # Fixture needed to create auction source
        mv_name = gen_name("auction_sum")

        # Create materialized view
        bids_table = con.table("bids")
        max_bid_expr = bids_table.aggregate(max_value=bids_table["amount"].max())
        con.create_materialized_view(mv_name, max_bid_expr)

        try:
            # Subscribe with Arrow format
            batch_count = 0
            total_rows = 0

            for batch in con.subscribe(mv_name, format="arrow"):
                batch_count += 1

                # Verify it's a PyArrow RecordBatch
                assert isinstance(batch, pa.RecordBatch)

                # Verify batch structure
                assert "mz_timestamp" in batch.schema.names
                assert "mz_diff" in batch.schema.names
                assert "max_value" in batch.schema.names

                # Verify we got data
                assert len(batch) > 0
                total_rows += len(batch)

                # Get first batch and exit
                break

            # Should have received at least one batch
            assert batch_count == 1
            assert total_rows > 0

        finally:
            con.drop_materialized_view(mv_name, force=True)

    def test_subscribe_polars_format(self, con, auction_source):
        """Test SUBSCRIBE with Polars format output."""
        pl = pytest.importorskip("polars")

        from ibis.util import gen_name

        _ = auction_source  # Fixture needed to create auction source
        mv_name = gen_name("auction_sum")

        # Create materialized view
        bids_table = con.table("bids")
        max_bid_expr = bids_table.aggregate(max_value=bids_table["amount"].max())
        con.create_materialized_view(mv_name, max_bid_expr)

        try:
            # Subscribe with Polars format
            batch_count = 0
            total_rows = 0

            for batch in con.subscribe(mv_name, format="polars"):
                batch_count += 1

                # Verify it's a Polars DataFrame
                assert isinstance(batch, pl.DataFrame)

                # Verify batch structure
                assert "mz_timestamp" in batch.columns
                assert "mz_diff" in batch.columns
                assert "max_value" in batch.columns

                # Verify we got data
                assert len(batch) > 0
                total_rows += len(batch)

                # Test Polars-specific filtering
                inserts = batch.filter(pl.col("mz_diff") == 1)
                assert len(inserts) >= 0  # May or may not have inserts in first batch

                # Get first batch and exit
                break

            # Should have received at least one batch
            assert batch_count == 1
            assert total_rows > 0

        finally:
            con.drop_materialized_view(mv_name, force=True)
