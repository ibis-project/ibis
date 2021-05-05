import pytest

import ibis.expr.datatypes as dt

pa = pytest.importorskip('pyarrow')


@pytest.mark.parametrize(
    ('datatype', 'expected'),
    [
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
        (pa.timestamp('us'), dt.timestamp),
        (pa.timestamp('ns', 'UTC'), dt.Timestamp('UTC')),
        (pa.timestamp('us', 'Europe/Paris'), dt.Timestamp('Europe/Paris')),
    ],
    ids=lambda x: str(x),
)
def test_convert_arrow(datatype, expected):
    assert dt.dtype(datatype) == expected
