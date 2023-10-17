from __future__ import annotations

import datetime

import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt


@pytest.mark.parametrize(
    "value,expected",
    [
        param(5, "CAST(5 AS TINYINT)", id="int"),
        param(1.5, "CAST(1.5 AS DOUBLE)", id="float"),
        param(True, "TRUE", id="true"),
        param(False, "FALSE", id="false"),
    ],
)
def test_simple_literals(con, value, expected):
    expr = ibis.literal(value)
    result = con.compile(expr)
    assert result == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        param("simple", "'simple'", id="simple"),
        param("I can't", "'I can''t'", id="nested_quote"),
        param('An "escape"', """'An "escape"'""", id="nested_token"),
    ],
)
def test_string_literals(con, value, expected):
    expr = ibis.literal(value)
    result = con.compile(expr)
    assert result == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        param(
            datetime.timedelta(seconds=70),
            "INTERVAL '00 00:01:10.000000' DAY TO SECOND",
            id="70seconds",
        ),
        param(
            ibis.interval(months=50), "INTERVAL '04-02' YEAR TO MONTH", id="50months"
        ),
        param(ibis.interval(seconds=5), "INTERVAL '5' SECOND", id="5seconds"),
    ],
)
def test_translate_interval_literal(con, value, expected):
    expr = ibis.literal(value)
    result = con.compile(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("case", "dtype"),
    [
        param(datetime.datetime(2017, 1, 1, 4, 55, 59), dt.timestamp, id="datetime"),
        param(
            datetime.datetime(2017, 1, 1, 4, 55, 59, 1122),
            dt.timestamp,
            id="datetime_with_microseconds",
        ),
        param("2017-01-01 04:55:59", dt.timestamp, id="string_timestamp"),
        param(pd.Timestamp("2017-01-01 04:55:59"), dt.timestamp, id="timestamp"),
        param(datetime.time(4, 55, 59), dt.time, id="time"),
        param("04:55:59", dt.time, id="string_time"),
    ],
)
def test_literal_timestamp_or_time(con, snapshot, case, dtype):
    expr = ibis.literal(case, type=dtype)
    result = con.compile(expr)
    snapshot.assert_match(result, "out.sql")
