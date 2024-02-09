from __future__ import annotations

import pytest
from pytest import param

import ibis
from ibis.conftest import LINUX, MACOS, SANDBOXED

pytestmark = pytest.mark.examples

pytest.importorskip("pins")


@pytest.mark.skipif(
    (LINUX or MACOS) and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
)
@pytest.mark.notimpl(["pyspark", "exasol"])
@pytest.mark.notyet(["clickhouse", "druid", "impala", "mssql", "trino", "risingwave"])
@pytest.mark.parametrize(
    ("example", "columns"),
    [
        param(
            "wowah_locations_raw",
            ["Map_ID", "Location_Type", "Location_Name", "Game_Version"],
            id="parquet",
        ),
        param(
            "band_instruments",
            ["name", "plays"],
            id="csv",
        ),
        param(
            "AwardsManagers",
            ["player_id", "award_id", "year_id", "lg_id", "tie", "notes"],
            id="csv-all-null",
            marks=pytest.mark.notimpl(
                ["flink"],
                raises=TypeError,
                reason=(
                    "Unsupported type: NULL, "
                    "it is not supported yet in current python type system"
                ),
            ),
        ),
    ],
)
def test_load_examples(con, example, columns):
    t = getattr(ibis.examples, example).fetch(backend=con)
    assert t.columns == columns
    assert t.count().execute() > 0
