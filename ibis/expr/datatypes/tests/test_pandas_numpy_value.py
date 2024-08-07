from __future__ import annotations

from datetime import date, datetime

import pytest
from packaging.version import parse as vparse

import ibis.expr.datatypes as dt

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        (pd.Timedelta("5 hours"), dt.Interval(unit="h")),
        (pd.Timedelta("7 minutes"), dt.Interval(unit="m")),
        (pd.Timedelta("11 milliseconds"), dt.Interval(unit="ms")),
        (pd.Timedelta("17 nanoseconds"), dt.Interval(unit="ns")),
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


# str, pd.Timestamp, datetime, np.datetime64, numbers.Real
@pytest.mark.parametrize(
    ("value", "expected"),
    [
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
    pytest.importorskip("pyarrow")
    pandas_series = pd.Series(numpy_array)
    assert dt.infer(numpy_array) in expected_dtypes
    assert dt.infer(pandas_series) in expected_dtypes


def test_normalize_non_convertible_boolean():
    typ = dt.boolean
    value = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="Unable to normalize .+ to Boolean"):
        dt.normalize(typ, value)
