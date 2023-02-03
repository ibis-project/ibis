import pytest
from multipledispatch.conflict import ambiguities
from pytest import param

import ibis.expr.datatypes as dt
from ibis.backends.bigquery.datatypes import (
    ibis_type_to_bigquery_type,
    spread_type,
)


def test_no_ambiguities():
    ambs = ambiguities(ibis_type_to_bigquery_type.funcs)
    assert not ambs


@pytest.mark.parametrize(
    ("datatype", "expected"),
    [
        param(dt.float32, "FLOAT64", id="float32"),
        param(dt.float64, "FLOAT64", id="float64"),
        param(dt.uint8, "INT64", id="uint8"),
        param(dt.uint16, "INT64", id="uint16"),
        param(dt.uint32, "INT64", id="uint32"),
        param(dt.int8, "INT64", id="int8"),
        param(dt.int16, "INT64", id="int16"),
        param(dt.int32, "INT64", id="int32"),
        param(dt.int64, "INT64", id="int64"),
        param(dt.string, "STRING", id="string"),
        param(dt.Array(dt.int64), "ARRAY<INT64>", id="array<int64>"),
        param(dt.Array(dt.string), "ARRAY<STRING>", id="array<string>"),
        param(
            dt.Struct.from_tuples(
                [("a", dt.int64), ("b", dt.string), ("c", dt.Array(dt.string))]
            ),
            "STRUCT<a INT64, b STRING, c ARRAY<STRING>>",
            id="struct",
        ),
        param(dt.date, "DATE", id="date"),
        param(dt.timestamp, "TIMESTAMP", id="timestamp"),
        param(
            dt.Timestamp(timezone="US/Eastern"),
            "TIMESTAMP",
            marks=pytest.mark.xfail(
                raises=TypeError, reason="Not supported in BigQuery"
            ),
            id="timestamp_with_tz",
        ),
        param(
            "array<struct<a: string>>", "ARRAY<STRUCT<a STRING>>", id="array<struct>"
        ),
        param(dt.Decimal(38, 9), "NUMERIC", id="decimal-numeric"),
        param(dt.Decimal(76, 38), "BIGNUMERIC", id="decimal-bignumeric"),
    ],
)
def test_simple(datatype, expected):
    assert ibis_type_to_bigquery_type(datatype) == expected


@pytest.mark.parametrize("datatype", [dt.uint64, dt.Decimal(8, 3)])
def test_simple_failure_mode(datatype):
    with pytest.raises(TypeError):
        ibis_type_to_bigquery_type(datatype)


@pytest.mark.parametrize(
    ("type_", "expected"),
    [
        param(
            dt.int64,
            [dt.int64],
        ),
        param(
            dt.Array(dt.int64),
            [dt.int64, dt.Array(value_type=dt.int64)],
        ),
        param(
            dt.Struct.from_tuples([("a", dt.Array(dt.int64))]),
            [
                dt.int64,
                dt.Array(value_type=dt.int64),
                dt.Struct.from_tuples([('a', dt.Array(value_type=dt.int64))]),
            ],
        ),
    ],
)
def test_spread_type(type_, expected):
    assert list(spread_type(type_)) == expected
