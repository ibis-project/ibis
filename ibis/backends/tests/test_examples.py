from __future__ import annotations

import pytest
from pytest import param

import ibis
from ibis.backends.tests.errors import ClickHouseDatabaseError, PyODBCProgrammingError
from ibis.conftest import LINUX, MACOS, SANDBOXED

pytestmark = pytest.mark.examples

pytest.importorskip("pins")


@pytest.mark.skipif(
    (LINUX or MACOS) and SANDBOXED,
    reason="nix on linux cannot download duckdb extensions or data due to sandboxing",
)
@pytest.mark.notyet(
    ["athena", "databricks", "druid", "exasol", "impala", "risingwave", "trino"]
)
@pytest.mark.parametrize(
    ("example", "columns"),
    [
        param(
            "wowah_locations_raw",
            ["Map_ID", "Location_Type", "Location_Name", "Game_Version"],
            id="parquet",
        ),
        param("band_instruments", ["name", "plays"], id="csv"),
        param(
            "AwardsManagers",
            ["player_id", "award_id", "year_id", "lg_id", "tie", "notes"],
            id="csv-all-null",
            marks=[
                pytest.mark.notimpl(
                    ["flink"],
                    raises=TypeError,
                    reason=(
                        "Unsupported type: NULL, "
                        "it is not supported yet in current python type system"
                    ),
                ),
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=ClickHouseDatabaseError,
                    reason="All null doesn't have a corresponding column type in clickhouse",
                ),
            ],
        ),
        param(
            "penguins",
            [
                "species",
                "island",
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "sex",
                "year",
            ],
            id="has-null-integer-values",
            marks=pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError),
        ),
    ],
)
def test_load_examples(con, example, columns):
    t = getattr(ibis.examples, example).fetch(backend=con)
    assert t.columns == tuple(columns)
    assert t.count().execute() > 0
