from __future__ import annotations

import json

import pandas.testing as tm
import pytest

pytest.importorskip("pyspark")


def test_json_array(con):
    table = con.table("nested_json_table")
    table_pandas = table.execute()

    result = (
        table.mutate(
            version_detail=table["version_detail"]
            .cast("json")
            .unwrap_as(
                "array<struct<description: string, details: struct<major_version: int64, released: bool, accurate_rate: float64>>>"
            )
        )
        .execute()
        .reset_index(drop=True)
    )
    expected = table_pandas.assign(
        version_detail=table_pandas.version_detail.apply(json.loads),
    ).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)
