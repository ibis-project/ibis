from __future__ import annotations

import pytest
from pytest import param

import ibis.expr.datatypes as dt
from ibis.backends.flink.datatypes import FlinkType


@pytest.mark.parametrize(
    ("datatype", "expected"),
    [
        param(dt.float32, "FLOAT", id="float32"),
        param(dt.float64, "DOUBLE", id="float64"),
        param(dt.uint8, "TINYINT", id="uint8"),
        param(dt.uint16, "SMALLINT", id="uint16"),
        param(dt.uint32, "INT", id="uint32"),
        param(dt.int8, "TINYINT", id="int8"),
        param(dt.int16, "SMALLINT", id="int16"),
        param(dt.int32, "INT", id="int32"),
        param(dt.int64, "BIGINT", id="int64"),
        param(dt.string, "VARCHAR", id="string"),
        param(dt.date, "DATE", id="date"),
        param(dt.timestamp, "TIMESTAMP", id="datetime"),
        param(
            dt.Timestamp(timezone="UTC"),
            "TIMESTAMP",
            id="timestamp_with_utc_tz",
        ),
        param(
            dt.Timestamp(timezone="US/Eastern"),
            "TIMESTAMP",
            id="timestamp_with_other_tz",
        ),
    ],
)
def test_simple(datatype, expected):
    assert FlinkType.to_string(datatype) == expected
