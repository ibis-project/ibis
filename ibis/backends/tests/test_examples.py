from __future__ import annotations

import pytest

import ibis
from ibis.conftest import LINUX, MACOS, SANDBOXED

pytestmark = pytest.mark.examples


@pytest.mark.skipif(
    (LINUX or MACOS) and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
)
@pytest.mark.notimpl(["dask", "datafusion", "pyspark"])
@pytest.mark.notyet(["bigquery", "clickhouse", "druid", "impala", "mssql", "trino"])
@pytest.mark.parametrize(
    ("example", "columns"),
    [
        (
            "wowah_locations_raw",
            ["Map_ID", "Location_Type", "Location_Name", "Game_Version"],
        ),
        ("band_instruments", ["name", "plays"]),
        (
            "AwardsManagers",
            ["player_id", "award_id", "year_id", "lg_id", "tie", "notes"],
        ),
    ],
    ids=["parquet", "csv", "csv-all-null"],
)
def test_load_examples(con, example, columns):
    t = getattr(ibis.examples, example).fetch(backend=con)
    assert t.columns == columns
    assert t.count().execute() > 0
