from __future__ import annotations

import polars as pl
import pytest
from pytest import param

import ibis.expr.datatypes as dt
from ibis.backends.polars.datatypes import dtype_from_polars, dtype_to_polars


@pytest.mark.parametrize(
    ("ibis_dtype", "polars_type"),
    [
        param(dt.bool, pl.Boolean, id="bool"),
        param(dt.null, pl.Null, id="null"),
        param(dt.Array(dt.string), pl.List(pl.Utf8), id="array_string"),
        param(dt.string, pl.Utf8, id="string"),
        param(dt.binary, pl.Binary, id="binary"),
        param(dt.date, pl.Date, id="date"),
        param(dt.time, pl.Time, id="time"),
        param(dt.int8, pl.Int8, id="int8"),
        param(dt.int16, pl.Int16, id="int16"),
        param(dt.int32, pl.Int32, id="int32"),
        param(dt.int64, pl.Int64, id="int64"),
        param(dt.uint8, pl.UInt8, id="uint8"),
        param(dt.uint16, pl.UInt16, id="uint16"),
        param(dt.uint32, pl.UInt32, id="uint32"),
        param(dt.uint64, pl.UInt64, id="uint64"),
        param(dt.float32, pl.Float32, id="float32"),
        param(dt.float64, pl.Float64, id="float64"),
        param(dt.timestamp, pl.Datetime("ns", time_zone=None), id="timestamp"),
        param(
            dt.Timestamp("UTC"), pl.Datetime("ns", time_zone="UTC"), id="timestamp_tz"
        ),
        param(dt.Interval(unit="ms"), pl.Duration("ms"), id="interval_ms"),
        param(dt.Interval(unit="us"), pl.Duration("us"), id="interval_us"),
        param(dt.Interval(unit="ns"), pl.Duration("ns"), id="interval_ns"),
        param(
            dt.Struct(
                dict(a=dt.string, b=dt.Array(dt.Array(dt.Struct(dict(c=dt.float64)))))
            ),
            pl.Struct(
                [
                    pl.Field("a", pl.Utf8),
                    pl.Field(
                        "b", pl.List(pl.List(pl.Struct([pl.Field("c", pl.Float64)])))
                    ),
                ]
            ),
            id="struct",
        ),
    ],
)
def test_to_from_ibis_type(ibis_dtype, polars_type):
    assert dtype_to_polars(ibis_dtype) == polars_type
    assert dtype_from_polars(polars_type) == ibis_dtype


def test_categorical():
    assert dtype_from_polars(pl.Categorical()) == dt.string
