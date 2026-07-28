"""Type round-trips through the chDB backend (memtable in, Arrow out)."""

from __future__ import annotations

import datetime
from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest

import ibis


@pytest.mark.parametrize(
    ("sql", "family", "value"),
    [
        ("toInt8(-8)", "integer", -8),
        ("toInt32(-320000)", "integer", -320000),
        ("toInt64(-6400000000)", "integer", -6400000000),
        ("toUInt64(18446744073709551615)", "integer", 18446744073709551615),
        ("toFloat32(0.5)", "floating", 0.5),
        ("toFloat64(2.25)", "floating", 2.25),
        ("true", "boolean", True),
        ("'héllo宝'", "string", "héllo宝"),
        ("toDate('2026-07-22')", "date", pd.Timestamp("2026-07-22")),
        ("toDecimal64(123.4567, 4)", "decimal", Decimal("123.4567")),
        ("[1, 2, 3]", "array", [1, 2, 3]),
    ],
)
def test_scalar_type_roundtrip(mem, sql, family, value):
    expr = mem.sql(f"SELECT {sql} AS c")
    dtype = expr.schema()["c"]
    # chDB columns are non-nullable; compare the type family, not nullability.
    assert getattr(dtype, f"is_{family}")()
    assert mem.execute(expr)["c"].tolist() == [value]


def test_datetime_no_scale_becomes_timestamp(mem):
    # chDB emits scale-less DateTime as Arrow uint32 seconds; the converter
    # restores it to a proper timestamp.
    expr = mem.sql("SELECT toDateTime('2026-07-22 01:02:03') AS dt")
    assert expr.schema()["dt"].is_timestamp()
    assert mem.execute(expr)["dt"].tolist() == [
        datetime.datetime(2026, 7, 22, 1, 2, 3)
    ]


def test_datetime64_with_timezone(mem):
    expr = mem.sql(
        "SELECT toDateTime64('2026-07-22 01:02:03.456', 3, 'UTC') AS dt"
    )
    got = mem.execute(expr)["dt"].tolist()[0]
    assert got == datetime.datetime(
        2026, 7, 22, 1, 2, 3, 456000, tzinfo=datetime.timezone.utc
    )


def test_nullable_roundtrip(mem):
    t = ibis.memtable(
        pa.table(
            {"x": pa.array([1, None, 3], pa.int64()), "s": ["a", None, "c"]}
        )
    )
    res = mem.execute(t.order_by("x"))
    assert res["x"].tolist()[0] == 1
    assert res["s"].isna().sum() == 1


def test_array_column_roundtrip(mem):
    t = ibis.memtable(
        pa.table({"id": [1, 2], "vals": pa.array([[1, 2, 3], [4, 5]])})
    )
    res = mem.execute(t.mutate(n=t.vals.length()).order_by("id"))
    assert res["n"].tolist() == [3, 2]


def test_map_type(mem):
    expr = mem.sql("SELECT map('a', 1, 'b', 2) AS m")
    assert expr.schema()["m"].is_map()
    assert mem.execute(expr)["m"].tolist()[0] == {"a": 1, "b": 2}


def test_struct_type(mem):
    expr = mem.sql("SELECT tuple(1, 'x') AS t")
    assert expr.schema()["t"].is_struct()


def test_decimal_arithmetic(mem):
    t = ibis.memtable({"p": [Decimal("1.50"), Decimal("2.25")]})
    res = mem.execute(t.p.sum())
    assert res == Decimal("3.75")
