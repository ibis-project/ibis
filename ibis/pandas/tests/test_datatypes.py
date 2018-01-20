import pytest
import numpy as np
import pandas as pd

from ibis.compat import DatetimeTZDtype, CategoricalDtype
from ibis.expr import datatypes as dt


@pytest.mark.parametrize(('value', 'expected_dtype'), [
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
    (np.float32(5.5), dt.float),
    (np.float64(5.55), dt.float64),
    (np.float64(5.55), dt.double),
    (np.bool_(True), dt.boolean),
    (np.bool_(False), dt.boolean),
    (np.arange(5, dtype='int32'), dt.Array(dt.int32)),

    # pandas types
    (pd.Timestamp('2015-01-01 12:00:00', tz='US/Eastern'),
     dt.Timestamp('US/Eastern'))
])
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


@pytest.mark.parametrize(('numpy_dtype', 'ibis_dtype'), [
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
    (np.timedelta64, dt.interval)
])
def test_numpy_dtype(numpy_dtype, ibis_dtype):
    assert dt.dtype(np.dtype(numpy_dtype)) == ibis_dtype


@pytest.mark.parametrize(('pandas_dtype', 'ibis_dtype'), [
    (DatetimeTZDtype(tz='US/Eastern', unit='ns'), dt.Timestamp('US/Eastern')),
    (CategoricalDtype(), dt.Category())
])
def test_pandas_dtype(pandas_dtype, ibis_dtype):
    assert dt.dtype(pandas_dtype) == ibis_dtype
