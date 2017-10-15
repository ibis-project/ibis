import pytest
import pandas as pd


pytest.importorskip('clickhouse_driver')
pytestmark = pytest.mark.clickhouse


def test_column_types(alltypes):
    df = alltypes.execute()
    assert df.tinyint_col.dtype.name == 'int8'
    assert df.smallint_col.dtype.name == 'int16'
    assert df.int_col.dtype.name == 'int32'
    assert df.bigint_col.dtype.name == 'int64'
    assert df.float_col.dtype.name == 'float32'
    assert df.double_col.dtype.name == 'float64'
    assert pd.core.common.is_datetime64_dtype(df.timestamp_col.dtype)
