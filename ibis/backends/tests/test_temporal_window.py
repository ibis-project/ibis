from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
from ibis import _

# `window_by` compiles to engine-native TUMBLE/HOP table functions, which only
# the streaming-oriented backends implement.
pytestmark = pytest.mark.notimpl(
    [
        "athena",
        "bigquery",
        "clickhouse",
        "databricks",
        "datafusion",
        "db2",
        "druid",
        "duckdb",
        "exasol",
        "impala",
        "materialize",
        "mssql",
        "mysql",
        "oracle",
        "polars",
        "postgres",
        "singlestoredb",
        "snowflake",
        "sqlite",
        "trino",
    ],
    raises=com.OperationNotDefinedError,
)

# These run in batch mode, where the backends agree. Streaming execution differs
# on watermarks and on trailing windows that never flush; those cases live in
# the flink and pyspark suites.


def test_tumble_window_by_grouped_agg(alltypes):
    expr = (
        alltypes.window_by(alltypes.timestamp_col)
        .tumble(size=ibis.interval(days=10))
        .agg(by=["string_col"], avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "string_col", "avg"]
    assert result.shape == (740, 4)


def test_tumble_window_by_ungrouped_agg(alltypes):
    expr = (
        alltypes.window_by(alltypes.timestamp_col)
        .tumble(size=ibis.interval(days=10))
        .agg(avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "avg"]
    assert result.shape == (74, 3)


def test_hop_window_by_grouped_agg(alltypes):
    expr = (
        alltypes.window_by(alltypes.timestamp_col)
        .hop(size=ibis.interval(days=10), slide=ibis.interval(days=5))
        .agg(by=["string_col"], avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "string_col", "avg"]
    # halving the slide overlaps the windows, so each row lands in two of them
    assert result.shape == (1470, 4)


def test_hop_window_by_ungrouped_agg(alltypes):
    expr = (
        alltypes.window_by(alltypes.timestamp_col)
        .hop(size=ibis.interval(days=10), slide=ibis.interval(days=5))
        .agg(avg=_.float_col.mean())
    )
    result = expr.to_pandas()
    assert list(result.columns) == ["window_start", "window_end", "avg"]
    assert result.shape == (147, 3)
