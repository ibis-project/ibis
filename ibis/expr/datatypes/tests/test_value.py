from __future__ import annotations

import decimal
import enum
import json
from collections import OrderedDict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytz
from packaging.version import parse as vparse

import ibis.expr.datatypes as dt


class Foo(enum.Enum):
    a = 1
    b = 2


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        (None, dt.null),
        (False, dt.boolean),
        (True, dt.boolean),
        ("foo", dt.string),
        (b"fooblob", dt.binary),
        (date.today(), dt.date),
        (datetime.now(), dt.timestamp),
        (timedelta(days=3), dt.Interval(unit="D")),
        (pd.Timedelta("5 hours"), dt.Interval(unit="h")),
        (pd.Timedelta("7 minutes"), dt.Interval(unit="m")),
        (timedelta(seconds=9), dt.Interval(unit="s")),
        (pd.Timedelta("11 milliseconds"), dt.Interval(unit="ms")),
        (timedelta(microseconds=15), dt.Interval(unit="us")),
        (pd.Timedelta("17 nanoseconds"), dt.Interval(unit="ns")),
        # numeric types
        (5, dt.int8),
        (5, dt.int8),
        (127, dt.int8),
        (128, dt.int16),
        (32767, dt.int16),
        (32768, dt.int32),
        (2147483647, dt.int32),
        (2147483648, dt.int64),
        (-5, dt.int8),
        (-128, dt.int8),
        (-129, dt.int16),
        (-32769, dt.int32),
        (-2147483649, dt.int64),
        (1.5, dt.double),
        (decimal.Decimal(1.5), dt.decimal),
        # parametric types
        (list("abc"), dt.Array(dt.string)),
        (set("abc"), dt.Array(dt.string)),
        ({1, 5, 6}, dt.Array(dt.int8)),
        (frozenset(list("abc")), dt.Array(dt.string)),
        ([1, 2, 3], dt.Array(dt.int8)),
        ([1, 128], dt.Array(dt.int16)),
        ([1, 128, 32768], dt.Array(dt.int32)),
        ([1, 128, 32768, 2147483648], dt.Array(dt.int64)),
        ({"a": 1, "b": 2, "c": 3}, dt.Map(dt.string, dt.int8)),
        ({1: 2, 3: 4, 5: 6}, dt.Map(dt.int8, dt.int8)),
        (
            {"a": [1.0, 2.0], "b": [], "c": [3.0]},
            dt.Map(dt.string, dt.Array(dt.double)),
        ),
        (
            OrderedDict(
                [
                    ("a", 1),
                    ("b", list("abc")),
                    ("c", OrderedDict([("foo", [1.0, 2.0])])),
                ]
            ),
            dt.Struct.from_tuples(
                [
                    ("a", dt.int8),
                    ("b", dt.Array(dt.string)),
                    (
                        "c",
                        dt.Struct.from_tuples([("foo", dt.Array(dt.double))]),
                    ),
                ]
            ),
        ),
        (Foo.a, dt.Enum()),
        # numpy types
        (np.int8(5), dt.int8),
        (np.int16(-1), dt.int16),
        (np.int32(2), dt.int32),
        (np.int64(-5), dt.int64),
        (np.uint8(5), dt.uint8),
        (np.uint16(50), dt.uint16),
        (np.uint32(500), dt.uint32),
        (np.uint64(5000), dt.uint64),
        (np.float32(5.5), dt.float32),
        (np.float64(5.55), dt.float64),
        (np.bool_(True), dt.boolean),
        (np.bool_(False), dt.boolean),
        # pandas types
        (
            pd.Timestamp("2015-01-01 12:00:00", tz="US/Eastern"),
            dt.Timestamp("US/Eastern"),
        ),
    ],
)
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


def test_infer_mixed_type_fails():
    data = [1, "a"]
    with pytest.raises(TypeError):
        dt.infer(data)


def test_infer_timestamp_with_tz():
    now_raw = datetime.utcnow()
    now_utc = pytz.utc.localize(now_raw)
    assert now_utc.tzinfo == pytz.UTC
    assert dt.infer(now_utc).timezone == str(pytz.UTC)


def test_infer_timedelta():
    assert dt.infer(timedelta(days=3)) == dt.Interval(unit="D")
    assert dt.infer(timedelta(hours=5)) == dt.Interval(unit="s")
    assert dt.infer(timedelta(minutes=7)) == dt.Interval(unit="s")
    assert dt.infer(timedelta(seconds=9)) == dt.Interval(unit="s")
    assert dt.infer(timedelta(milliseconds=11)) == dt.Interval(unit="us")
    assert dt.infer(timedelta(microseconds=13)) == dt.Interval(unit="us")

    msg = "Unable to infer interval type from zero value"
    with pytest.raises(ValueError, match=msg):
        dt.infer(timedelta(days=0, seconds=0))

    msg = "Unable to infer interval type from mixed units"
    with pytest.raises(ValueError, match=msg):
        dt.infer(timedelta(days=1, hours=2))
    with pytest.raises(ValueError, match=msg):
        dt.infer(timedelta(days=1, microseconds=2))


# str, pd.Timestamp, datetime, np.datetime64, numbers.Real
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019-01-01", datetime(2019, 1, 1)),
        ("2019-01-01 00:00:00", datetime(2019, 1, 1)),
        ("2019-01-01 01:02:03.000004", datetime(2019, 1, 1, 1, 2, 3, 4)),
        (
            "2019-01-01 01:02:03.000004+00:00",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.utc),
        ),
        (
            "2019-01-01 01:02:03.000004+01:00",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.FixedOffset(60)),
        ),
        (
            "2019-01-01 01:02:03.000004-01:00",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.FixedOffset(-60)),
        ),
        (
            "2019-01-01 01:02:03.000004+01",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.FixedOffset(60)),
        ),
        (datetime(2019, 1, 1), datetime(2019, 1, 1)),
        (datetime(2019, 1, 1, 1, 2, 3, 4), datetime(2019, 1, 1, 1, 2, 3, 4)),
        (pd.Timestamp("2019-01-01"), datetime(2019, 1, 1)),
        (pd.Timestamp("2019-01-01 00:00:00"), datetime(2019, 1, 1)),
        (pd.Timestamp("2019-01-01 01:02:03.000004"), datetime(2019, 1, 1, 1, 2, 3, 4)),
        (np.datetime64("2019-01-01"), datetime(2019, 1, 1)),
        (np.datetime64("2019-01-01 01:02:03"), datetime(2019, 1, 1, 1, 2, 3)),
    ],
)
def test_normalize_timestamp(value, expected):
    normalized = dt.normalize(dt.timestamp, value)
    assert normalized == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019-01-01", date(2019, 1, 1)),
        ("2019-01-01 00:00:00", date(2019, 1, 1)),
        ("2019-01-01 01:02:03.000004", date(2019, 1, 1)),
        (datetime(2019, 1, 1), date(2019, 1, 1)),
        (datetime(2019, 1, 1, 1, 2, 3, 4), date(2019, 1, 1)),
        (pd.Timestamp("2019-01-01"), date(2019, 1, 1)),
        (pd.Timestamp("2019-01-01 00:00:00"), date(2019, 1, 1)),
        (pd.Timestamp("2019-01-01 01:02:03.000004"), date(2019, 1, 1)),
        (np.datetime64("2019-01-01"), date(2019, 1, 1)),
        (np.datetime64("2019-01-01 01:02:03"), date(2019, 1, 1)),
    ],
)
def test_normalize_date(value, expected):
    normalized = dt.normalize(dt.date, value)
    assert normalized == expected


@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        (dt.Interval("s"), timedelta(seconds=1), 1),
        (dt.Interval("ms"), timedelta(seconds=1), 1000),
        (dt.Interval("us"), timedelta(seconds=1), 1000000),
        (dt.Interval("ns"), timedelta(seconds=1), 1000000000),
    ],
)
def test_normalize_interval(dtype, value, expected):
    normalized = dt.normalize(dtype, value)
    assert normalized == expected


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        # numpy types
        (np.int8(5), dt.int8),
        (np.int16(-1), dt.int16),
        (np.int32(2), dt.int32),
        (np.int64(-5), dt.int64),
        (np.uint8(5), dt.uint8),
        (np.uint16(50), dt.uint16),
        (np.uint32(500), dt.uint32),
        (np.uint64(5000), dt.uint64),
        (np.float32(5.5), dt.float32),
        (np.float64(5.55), dt.float64),
        (np.bool_(True), dt.boolean),
        (np.bool_(False), dt.boolean),
        # pandas types
        (
            pd.Timestamp("2015-01-01 12:00:00", tz="US/Eastern"),
            dt.Timestamp("US/Eastern"),
        ),
    ],
)
def test_infer_numpy_scalar(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


@pytest.mark.parametrize(
    ("numpy_dtype", "ibis_dtype"),
    [
        (np.bool_, dt.boolean),
        (np.int8, dt.int8),
        (np.int16, dt.int16),
        (np.int32, dt.int32),
        (np.int64, dt.int64),
        (np.uint8, dt.uint8),
        (np.uint16, dt.uint16),
        (np.uint32, dt.uint32),
        (np.uint64, dt.uint64),
        (np.float16, dt.float16),
        (np.float32, dt.float32),
        (np.float64, dt.float64),
        (np.double, dt.double),
        (np.str_, dt.string),
        (np.datetime64, dt.timestamp),
    ],
)
def test_from_numpy_dtype(numpy_dtype, ibis_dtype):
    numpy_dtype = np.dtype(numpy_dtype)
    assert dt.DataType.from_numpy(numpy_dtype) == ibis_dtype
    assert dt.dtype(numpy_dtype) == ibis_dtype


def test_from_numpy_timedelta():
    if vparse(pytest.importorskip("pyarrow").__version__) < vparse("9"):
        pytest.skip("pyarrow < 9 globally mutates the timedelta64 numpy dtype")

    numpy_dtype = np.dtype(np.timedelta64)
    assert dt.DataType.from_numpy(numpy_dtype) == dt.Interval("s")
    assert dt.dtype(numpy_dtype) == dt.Interval("s")


@pytest.mark.parametrize(
    ("numpy_array", "expected_dtypes"),
    [
        # Explicitly-defined dtype
        (np.array([1, 2, 3], dtype="int8"), (dt.Array(dt.int8),)),
        (np.array([1, 2, 3], dtype="int16"), (dt.Array(dt.int16),)),
        (np.array([1, 2, 3], dtype="int32"), (dt.Array(dt.int32),)),
        (np.array([1, 2, 3], dtype="int64"), (dt.Array(dt.int64),)),
        (np.array([1, 2, 3], dtype="uint8"), (dt.Array(dt.uint8),)),
        (np.array([1, 2, 3], dtype="uint16"), (dt.Array(dt.uint16),)),
        (np.array([1, 2, 3], dtype="uint32"), (dt.Array(dt.uint32),)),
        (np.array([1, 2, 3], dtype="uint64"), (dt.Array(dt.uint64),)),
        (np.array([1.0, 2.0, 3.0], dtype="float32"), (dt.Array(dt.float32),)),
        (np.array([1.0, 2.0, 3.0], dtype="float64"), (dt.Array(dt.float64),)),
        (np.array([True, False, True], dtype="bool"), (dt.Array(dt.boolean),)),
        # Implicit dtype
        # Integer array could be inferred to int64 or int32 depending on system
        (np.array([1, 2, 3]), (dt.Array(dt.int64), dt.Array(dt.int32))),
        (np.array([1.0, 2.0, 3.0]), (dt.Array(dt.float64),)),
        (np.array([np.nan, np.nan, np.nan]), (dt.Array(dt.float64),)),
        (np.array([True, False, True]), (dt.Array(dt.boolean),)),
        (np.array(["1", "2", "3"]), (dt.Array(dt.string),)),
        (
            np.array(
                [
                    pd.Timestamp("2015-01-01 12:00:00"),
                    pd.Timestamp("2015-01-02 12:00:00"),
                    pd.Timestamp("2015-01-03 12:00:00"),
                ]
            ),
            (dt.Array(dt.Timestamp()), dt.Array(dt.Timestamp(scale=6))),
        ),
        # Implied from object dtype
        (np.array([1, 2, 3], dtype=object), (dt.Array(dt.int64),)),
        (np.array([1.0, 2.0, 3.0], dtype=object), (dt.Array(dt.float64),)),
        (np.array([True, False, True], dtype=object), (dt.Array(dt.boolean),)),
        (np.array(["1", "2", "3"], dtype=object), (dt.Array(dt.string),)),
        (
            np.array(
                [
                    pd.Timestamp("2015-01-01 12:00:00"),
                    pd.Timestamp("2015-01-02 12:00:00"),
                    pd.Timestamp("2015-01-03 12:00:00"),
                ],
                dtype=object,
            ),
            (dt.Array(dt.Timestamp()), dt.Array(dt.Timestamp(scale=6))),
        ),
    ],
)
def test_infer_numpy_array(numpy_array, expected_dtypes):
    pandas_series = pd.Series(numpy_array)
    assert dt.infer(numpy_array) in expected_dtypes
    assert dt.infer(pandas_series) in expected_dtypes


def test_normalize_json():
    obj = ["foo", {"bar": ("baz", None, 1.0, 2)}]
    expected = json.dumps(obj)

    assert dt.normalize(dt.json, obj) == expected
    assert dt.normalize(dt.json, expected) == expected

    with pytest.raises(TypeError):
        dt.normalize(dt.json, "invalid")


def test_normalize_none_with_non_nullable_type():
    typ = dt.Int64(nullable=False)
    with pytest.raises(TypeError, match="Cannot convert `None` to non-nullable type"):
        dt.normalize(typ, None)


def test_normalize_non_convertible_boolean():
    typ = dt.boolean
    value = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="Unable to normalize .+ to Boolean"):
        dt.normalize(typ, value)


@pytest.mark.parametrize("bits", [8, 16, 32, 64])
@pytest.mark.parametrize("kind", ["uint", "int"])
def test_normalize_non_convertible_int(kind, bits):
    typ = getattr(dt, f"{kind}{bits:d}")
    with pytest.raises(TypeError, match="Unable to normalize .+ to U?Int"):
        dt.normalize(typ, "not convertible")


@pytest.mark.parametrize("typename", ["float32", "float64"])
def test_normalize_non_convertible_float(typename):
    typ = getattr(dt, typename)
    with pytest.raises(TypeError, match="Unable to normalize .+ to Float"):
        dt.normalize(typ, "not convertible")


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (1, dt.Decimal(), "1"),
        (1.0, dt.Decimal(), "1"),
        (1.0, dt.Decimal(2, 1), "1.0"),
        (1.0, dt.Decimal(2, 0), "1"),
        (1.0, dt.Decimal(4, 3), "1.000"),
        (12, dt.Decimal(6, 3), "12.000"),
        (12.1234, dt.Decimal(7, 5), "12.12340"),
        (True, dt.Decimal(4, 0), "1"),
        (True, dt.Decimal(4, 3), "1.000"),
        (False, dt.Decimal(4, 0), "0"),
        (decimal.Decimal("1.1"), dt.Decimal(76, 38), "1.1" + "0" * 37),
    ],
)
def test_normalize_decimal(value, dtype, expected):
    assert str(dt.normalize(dtype, value)) == expected


def test_normalize_decimal_invalid():
    with pytest.raises(TypeError):
        dt.normalize(dt.Decimal(4, 2), "invalid")
    with pytest.raises(TypeError):
        dt.normalize(dt.Decimal(4, 2), 1234)
    with pytest.raises(TypeError):
        dt.normalize(12.1234, dt.Decimal(6, 2))
