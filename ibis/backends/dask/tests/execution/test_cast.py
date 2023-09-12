from __future__ import annotations

import decimal

import pytest
import pytz
from pandas import Timestamp
from pytest import param

import ibis
import ibis.expr.datatypes as dt

pytest.importorskip("dask.dataframe")
from dask.dataframe.utils import tm  # noqa: E402

from ibis.backends.dask.execution import execute  # noqa: E402

TIMESTAMP = "2022-03-13 06:59:10.467417"


@pytest.mark.parametrize("from_", ["plain_float64", "plain_int64"])
@pytest.mark.parametrize(
    ("to", "expected"),
    [
        ("float16", "float16"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("double", "float64"),
        ("float", "float64"),
        ("int8", "int8"),
        ("int16", "int16"),
        ("int32", "int32"),
        ("int64", "int64"),
        ("string", "object"),
    ],
)
def test_cast_numeric(t, df, from_, to, expected):
    c = t[from_].cast(to)
    result = c.execute()
    assert str(result.dtype) == expected


@pytest.mark.parametrize("from_", ["float64_as_strings", "int64_as_strings"])
@pytest.mark.parametrize(
    ("to", "expected"), [("double", "float64"), ("string", "object")]
)
def test_cast_string(t, df, from_, to, expected):
    c = t[from_].cast(to)
    result = c.execute()
    assert str(result.dtype) == expected


@pytest.mark.parametrize(
    ("to", "expected"),
    [
        ("string", "object"),
        ("int64", "int64"),
        param("double", "float64", marks=pytest.mark.xfail(raises=TypeError)),
        (
            dt.Timestamp("America/Los_Angeles"),
            "datetime64[ns, America/Los_Angeles]",
        ),
        (
            "timestamp('America/Los_Angeles')",
            "datetime64[ns, America/Los_Angeles]",
        ),
    ],
)
@pytest.mark.parametrize(
    "column",
    ["plain_datetimes_naive", "plain_datetimes_ny", "plain_datetimes_utc"],
)
def test_cast_timestamp_column(t, df, column, to, expected):
    c = t[column].cast(to)
    result = c.execute()
    assert str(result.dtype) == expected


@pytest.mark.parametrize(
    ("to", "expected"),
    [
        ("string", str),
        ("int64", lambda x: Timestamp(x).value // int(1e9)),
        param(
            "double",
            float,
            marks=pytest.mark.notimpl(["dask"]),
        ),
        (
            dt.Timestamp("America/Los_Angeles"),
            lambda x: x.astimezone(tz=pytz.timezone("America/Los_Angeles")),
        ),
    ],
)
def test_cast_timestamp_scalar_naive(to, expected):
    literal_expr = ibis.literal(Timestamp(TIMESTAMP))
    value = literal_expr.cast(to)
    result = execute(value.op())
    raw = execute(literal_expr.op())
    assert result == expected(raw)


@pytest.mark.parametrize(
    ("to", "expected"),
    [
        ("string", str),
        ("int64", lambda x: Timestamp(x).value // int(1e9)),
        param("double", float, marks=pytest.mark.notimpl(["dask"])),
        (
            dt.Timestamp("America/Los_Angeles"),
            lambda x: x.astimezone(tz=pytz.timezone("America/Los_Angeles")),
        ),
    ],
)
@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
def test_cast_timestamp_scalar(to, expected, tz):
    literal_expr = ibis.literal(Timestamp(TIMESTAMP).tz_localize(tz))
    value = literal_expr.cast(to)
    result = execute(value.op())
    raw = execute(literal_expr.op())
    assert result == expected(raw)


def test_timestamp_with_timezone_is_inferred_correctly(t):
    assert t.plain_datetimes_naive.type().equals(dt.timestamp)
    assert t.plain_datetimes_ny.type().equals(dt.Timestamp("America/New_York"))
    assert t.plain_datetimes_utc.type().equals(dt.Timestamp("UTC"))


@pytest.mark.parametrize(
    "column",
    ["plain_datetimes_naive", "plain_datetimes_ny", "plain_datetimes_utc"],
)
def test_cast_date(t, df, column):
    expr = t[column].cast("date")
    result = expr.compile()
    expected = df[column].dt.normalize()
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


@pytest.mark.parametrize("type", [dt.Decimal(9, 0), dt.Decimal(12, 3)])
def test_cast_to_decimal(t, df, type):
    expr = t.float64_as_strings.cast(type)
    result = expr.compile()
    context = decimal.Context(prec=type.precision)
    expected = df.float64_as_strings.apply(
        lambda x: context.create_decimal(x).quantize(
            decimal.Decimal(
                "{}.{}".format("0" * (type.precision - type.scale), "0" * type.scale)
            )
        ),
        meta=("float64_as_strings", "object"),
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)
    assert all(
        abs(element.as_tuple().exponent) == type.scale
        for element in result.compute().values
    )
    assert all(
        1 <= len(element.as_tuple().digits) <= type.precision
        for element in result.compute().values
    )
