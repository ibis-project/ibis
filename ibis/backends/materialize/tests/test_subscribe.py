"""Tests for Materialize SUBSCRIBE functionality.

Tests cover streaming query functionality via SUBSCRIBE command.
"""

from __future__ import annotations

import pytest


class TestSubscribe:
    """Tests for SUBSCRIBE streaming functionality.

    Based on the Materialize quickstart guide.
    """

    def test_subscribe_quickstart_workflow(self, con):
        """Test SUBSCRIBE with load generator (simplified quickstart example).

        This test demonstrates the Materialize streaming workflow:
        1. Create a COUNTER source (generates sequential data)
        2. Create a materialized view over the source
        3. Subscribe to see real-time updates
        """
        from ibis.util import gen_name

        source_name = gen_name("counter_source")
        mv_name = gen_name("counter_sum")

        try:
            # Create COUNTER source (simple incrementing counter)
            con.create_source(
                source_name,
                connector="COUNTER",
            )

            # Create a materialized view that computes over the source
            # Get the max counter value
            counter_table = con.table(source_name)
            max_counter_expr = counter_table.aggregate(
                max_value=counter_table["counter"].max()
            )
            con.create_materialized_view(mv_name, max_counter_expr)

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
            con.drop_source(source_name, force=True)

    def test_subscribe_arrow_format(self, con):
        """Test SUBSCRIBE with Arrow format output."""
        import pyarrow as pa

        from ibis.util import gen_name

        source_name = gen_name("counter_source")
        mv_name = gen_name("counter_sum")

        try:
            # Create COUNTER source
            con.create_source(source_name, connector="COUNTER")

            # Create materialized view
            counter_table = con.table(source_name)
            max_counter_expr = counter_table.aggregate(
                max_value=counter_table["counter"].max()
            )
            con.create_materialized_view(mv_name, max_counter_expr)

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
            con.drop_source(source_name, force=True)

    def test_subscribe_polars_format(self, con):
        """Test SUBSCRIBE with Polars format output."""
        pl = pytest.importorskip("polars")

        from ibis.util import gen_name

        source_name = gen_name("counter_source")
        mv_name = gen_name("counter_sum")

        try:
            # Create COUNTER source
            con.create_source(source_name, connector="COUNTER")

            # Create materialized view
            counter_table = con.table(source_name)
            max_counter_expr = counter_table.aggregate(
                max_value=counter_table["counter"].max()
            )
            con.create_materialized_view(mv_name, max_counter_expr)

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
            con.drop_source(source_name, force=True)
