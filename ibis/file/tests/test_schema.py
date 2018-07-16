import tempfile
import pytest
import numpy as np
import pandas as pd
import ibis
import ibis.expr.datatypes as dt
from ibis.compat import PY2

pa = pytest.importorskip('pyarrow')  # noqa: E402
import pyarrow.parquet as pq


@pytest.mark.parametrize(('datatype', 'expected'), [
    (pa.int8(), dt.int8),
    (pa.int16(), dt.int16),
    (pa.int32(), dt.int32),
    (pa.int64(), dt.int64),
    (pa.uint8(), dt.uint8),
    (pa.uint16(), dt.uint16),
    (pa.uint32(), dt.uint32),
    (pa.uint64(), dt.uint64),
    (pa.float16(), dt.float16),
    (pa.float32(), dt.float32),
    (pa.float64(), dt.float64),
    (pa.string(), dt.string),
    (pa.timestamp('ns'), dt.timestamp),
], ids=lambda x: str(x))
def test_convert_arrow(datatype, expected):
    assert dt.dtype(datatype) == expected


@pytest.fixture
def parquet_schema():
    np.random.seed(0)
    size = 100
    df = pd.DataFrame({
        'uint8': np.arange(size, dtype=np.uint8),
        'uint16': np.arange(size, dtype=np.uint16),
        'uint32': np.arange(size, dtype=np.uint32),
        'uint64': np.arange(size, dtype=np.uint64),
        'int8': np.arange(size, dtype=np.int16),
        'int16': np.arange(size, dtype=np.int16),
        'int32': np.arange(size, dtype=np.int32),
        'int64': np.arange(size, dtype=np.int64),
        'float32': np.arange(size, dtype=np.float32),
        'float64': np.arange(size, dtype=np.float64),
        'bool': np.random.randn(size) > 0,
        # TODO(wesm): Test other timestamp resolutions now that arrow supports
        # them
        'datetime': np.arange("2016-01-01T00:00:00.001", size,
                              dtype='datetime64[ms]'),
        'str': [str(x) for x in range(size)],
        'str_with_nulls': [None] + [str(x) for x in range(size - 2)] + [None],
        'empty_str': [''] * size,
        'bytes': [b'foo'] * size
        }, columns=['uint8', 'uint16', 'uint32', 'uint64',
                    'int8', 'int16', 'int32',
                    'int64', 'float32', 'float64', 'bool',
                    'datetime', 'str', 'str_with_nulls', 'empty_str',
                    'bytes'])

    with tempfile.TemporaryFile() as path:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        parquet_file = pq.ParquetFile(path)
        return parquet_file.schema


def test_convert_parquet(parquet_schema):
    # TODO(jreback)
    # not entirely sure this is correct
    # should these be strings in py2?
    if PY2:
        strings = [dt.binary, dt.binary, dt.binary]
    else:
        strings = [dt.string, dt.string, dt.string]

    # uint32, int8, int16 stored as upcasted types
    types = [dt.uint8, dt.uint16, dt.int64, dt.uint64,
             dt.int16, dt.int16, dt.int32,
             dt.int64, dt.float32, dt.float64, dt.boolean,
             dt.timestamp] + strings + [dt.binary, dt.int64]
    names = ['uint8', 'uint16', 'uint32', 'uint64',
             'int8', 'int16', 'int32',
             'int64', 'float32', 'float64', 'bool',
             'datetime', 'str', 'str_with_nulls', 'empty_str',
             'bytes']
    expected = ibis.schema(zip(names, types))

    result = ibis.infer_schema(parquet_schema)
    assert result == expected
