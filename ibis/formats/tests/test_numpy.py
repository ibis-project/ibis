from __future__ import annotations

import hypothesis as h
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from packaging.version import parse as vparse

import ibis.expr.datatypes as dt
import ibis.tests.strategies as ibst
from ibis.formats.numpy import NumpySchema, NumpyType

roundtripable_types = st.deferred(
    lambda: (
        npst.integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.datetime64_dtypes(max_period="ns", endianness="=")
        | npst.timedelta64_dtypes(max_period="s", endianness="=")
    )
)


@st.composite
def numpy_schema(draw, item_strategy=roundtripable_types, max_size=10):
    num_fields = draw(st.integers(min_value=0, max_value=max_size))
    names = draw(
        st.lists(st.text(), min_size=num_fields, max_size=num_fields, unique=True)
    )
    types = draw(st.lists(item_strategy, min_size=num_fields, max_size=num_fields))
    return list(zip(names, types))


def assert_dtype_roundtrip(
    numpy_type, ibis_type=None, restored_type=None, nullable=True
):
    dtype = NumpyType.to_ibis(numpy_type, nullable=nullable)
    if ibis_type is not None:
        assert dtype == ibis_type

    nptyp = NumpyType.from_ibis(dtype)
    if restored_type is None:
        restored_type = numpy_type
    assert nptyp == restored_type


@h.given(roundtripable_types)
def test_roundtripable_types(numpy_type):
    assert_dtype_roundtrip(numpy_type, nullable=False)
    assert_dtype_roundtrip(numpy_type, nullable=True)


@h.given(npst.unicode_string_dtypes(endianness="="))
def test_non_roundtripable_str_type(numpy_type):
    assert_dtype_roundtrip(
        numpy_type, dt.String(nullable=False), np.dtype("object"), nullable=False
    )
    assert_dtype_roundtrip(
        numpy_type, dt.String(nullable=True), np.dtype("object"), nullable=True
    )


@h.given(npst.byte_string_dtypes(endianness="="))
def test_non_roundtripable_bytes_type(numpy_type):
    assert_dtype_roundtrip(
        numpy_type, dt.Binary(nullable=False), np.dtype("object"), nullable=False
    )
    assert_dtype_roundtrip(
        numpy_type, dt.Binary(nullable=True), np.dtype("object"), nullable=True
    )


@h.given(
    ibst.null_dtype
    | ibst.variadic_dtypes()
    | ibst.decimal_dtypes()
    | ibst.struct_dtypes()
)
def test_variadic_to_numpy(ibis_type):
    assert NumpyType.from_ibis(ibis_type) == np.dtype("object")


@h.given(ibst.date_dtype() | ibst.timestamp_dtype())
def test_date_to_numpy(ibis_type):
    assert NumpyType.from_ibis(ibis_type) == np.dtype("datetime64[ns]")


@h.given(ibst.time_dtype())
def test_time_to_numpy(ibis_type):
    assert NumpyType.from_ibis(ibis_type) == np.dtype("timedelta64[ns]")


@h.given(ibst.schema())
def test_schema_to_numpy(ibis_schema):
    numpy_schema = NumpySchema.from_ibis(ibis_schema)
    assert len(numpy_schema) == len(ibis_schema)

    for name, numpy_type in numpy_schema:
        assert numpy_type == NumpyType.from_ibis(ibis_schema[name])


@h.given(numpy_schema())
def test_schema_from_numpy(numpy_schema):
    ibis_schema = NumpySchema.to_ibis(numpy_schema)
    assert len(numpy_schema) == len(ibis_schema)

    for name, numpy_type in numpy_schema:
        assert NumpyType.from_ibis(ibis_schema[name]) == numpy_type


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
def test_dtype_from_numpy(numpy_dtype, ibis_dtype):
    assert NumpyType.to_ibis(np.dtype(numpy_dtype)) == ibis_dtype


def test_dtype_from_numpy_dtype_timedelta():
    if vparse(pytest.importorskip("pyarrow").__version__) < vparse("9"):
        pytest.skip("pyarrow < 9 globally mutates the timedelta64 numpy dtype")

    assert NumpyType.to_ibis(np.dtype(np.timedelta64)) == dt.Interval(unit="s")
