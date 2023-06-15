from __future__ import annotations

import datetime
import decimal

import pandas as pd
import pytest
from rich.console import Console

import ibis
import ibis.expr.datatypes as dt
from ibis.expr.types.pretty import format_column, format_values

null = "NULL"


def test_format_int_column():
    values = [None, 1, 2, 3, None]
    fmts, min_len, max_len = format_column(dt.int64, values)
    strs = [str(f) for f in fmts]
    assert strs == [null, "1", "2", "3", null]
    assert min_len == 4
    assert max_len == 4


def test_format_bool_column():
    values = [None, True, False]
    fmts, _, _ = format_column(dt.bool, values)
    strs = [str(f) for f in fmts]
    assert strs == [null, "True", "False"]


def test_format_float_column():
    values = [float("nan"), 1.52, -2.0, 0.0, float("inf"), float("-inf"), None]
    fmts, _, _ = format_column(dt.float64, values)
    strs = [str(f) for f in fmts]
    # matching trailing zeros are stripped
    assert strs == ["nan", "1.52", "-2.00", "0.00", "inf", "-inf", null]


def test_format_big_float_column():
    values = [1.52e6, -2.0e-6]
    fmts, _, _ = format_column(dt.float64, values)
    strs = [str(f) for f in fmts]
    assert strs == ["1.520000e+06", "-2.000000e-06"]


@pytest.mark.parametrize("is_float", [False, True])
def test_format_decimal(is_float):
    values = [decimal.Decimal("1.500"), decimal.Decimal("2.510")]
    if is_float:
        values = [float(v) for v in values]

    # With a scale, render using specified scale, even if backend
    # doesn't return results as a `Decimal` object.
    fmts, _, _ = format_column(dt.Decimal(scale=3), values)
    strs = [str(f) for f in fmts]
    assert strs == ["1.500", "2.510"]

    # Without scale, decimals render same as floats
    fmts, _, _ = format_column(dt.Decimal(scale=None), values)
    strs = [str(f) for f in fmts]
    assert strs == ["1.50", "2.51"]


@pytest.mark.parametrize(
    "t, prec",
    [
        ("2022-02-02 01:02:03", "seconds"),
        ("2022-02-02 01:02:03.123", "milliseconds"),
        ("2022-02-02 01:02:03.123456", "microseconds"),
    ],
)
def test_format_timestamp_column(t, prec):
    a = pd.Timestamp("2022-02-02")
    b = pd.Timestamp(t)
    NaT = pd.Timestamp("NaT")
    fmts, _, _ = format_column(dt.timestamp, [a, b, NaT])
    strs = [str(f) for f in fmts]
    assert strs == [
        a.isoformat(sep=" ", timespec=prec),
        b.isoformat(sep=" ", timespec=prec),
        null,
    ]


@pytest.mark.parametrize(
    "t, prec",
    [
        ("01:02:03", "seconds"),
        ("01:02:03.123", "milliseconds"),
        ("01:02:03.123456", "microseconds"),
    ],
)
def test_format_time_column(t, prec):
    a = datetime.time(4, 5, 6)
    b = datetime.time.fromisoformat(t)
    NaT = pd.Timestamp("NaT")
    fmts, _, _ = format_column(dt.time, [a, b, NaT])
    strs = [str(f) for f in fmts]
    assert strs == [
        a.isoformat(timespec=prec),
        b.isoformat(timespec=prec),
        null,
    ]


@pytest.mark.parametrize("is_date", [True, False])
def test_format_date_column(is_date):
    cls = datetime.date if is_date else datetime.datetime
    values = [cls(2022, 1, 1), cls(2022, 1, 2)]
    fmts, _, _ = format_column(dt.date, values)
    strs = [str(f) for f in fmts]
    assert strs == ["2022-01-01", "2022-01-02"]


def test_format_interval_column():
    values = [datetime.timedelta(seconds=1)]
    fmts, _, _ = format_column(dt.Interval("s"), values)
    strs = [str(f) for f in fmts]
    assert strs == [str(v) for v in values]


def test_format_string_column():
    max_string = ibis.options.repr.interactive.max_string
    values = [None, "", "test\t\r\n\v\f", "a string", "x" * (max_string + 10)]
    fmts, min_len, max_len = format_column(dt.string, values)
    strs = [str(f) for f in fmts]
    assert strs == [
        null,
        "~",
        "test\\t\\r\\n\\v\\f",
        "a string",
        "x" * (max_string - 1) + "…",
    ]
    assert min_len == 20
    assert max_len == max(map(len, strs))


def test_format_short_string_column():
    values = [None, "", "ab", "cd"]
    fmts, min_len, max_len = format_column(dt.string, values)
    strs = [str(f) for f in fmts]
    assert strs == [null, "~", "ab", "cd"]
    assert min_len == 4
    assert max_len == 4


def test_format_struct_column():
    dtype = dt.Struct({"x": "int", "y": "float"})
    values = [{"x": 1, "y": 2.5}, None]
    fmts, min_len, max_len = format_column(dtype, values)
    assert str(fmts[1]) == null
    assert min_len == 20
    assert max_len is None


def test_format_map_column():
    dtype = dt.Map(dt.String(), dt.Int32())
    values = [[("x", 1), ("y", 2.5)], None]
    fmts, min_len, max_len = format_column(dtype, values)
    assert str(fmts[1]) == null
    assert min_len == 20
    assert max_len is None


def test_format_fully_null_column():
    values = [None, None, None]
    fmts, *_ = format_column(dt.int64, values)
    strs = [str(f) for f in fmts]
    assert strs == [null, null, null]


def test_all_empty_groups_repr():
    values = [float("nan"), float("nan")]
    dtype = dt.float64
    fmts, *_ = format_column(dtype, values)
    assert list(map(str, fmts)) == ["nan", "nan"]


def test_non_interactive_column_repr():
    t = ibis.table(dict(names="string", values="int"))
    expr = t.names
    console = Console()
    with console.capture() as capture:
        console.print(expr)

    result = capture.get()
    assert "Selection" not in result


def test_format_link():
    result = format_values(dt.string, ["https://ibis-project.org"])[0]
    assert str(result) == "https://ibis-project.org"
    assert result.spans[0].style == "link https://ibis-project.org"

    # Check that long urls are truncated visually, but the link
    # itself is still the full url.
    long = "https://" + "x" * 1000
    result = format_values(dt.string, [long])[0]
    assert str(result).startswith("https://xxxxx")
    assert len(str(result)) < 1000
    assert result.spans[0].style == "link " + long

    # not links
    results = format_values(dt.string, ["https://", "https:", "https"])
    assert all(not rendered.spans for rendered in results)
