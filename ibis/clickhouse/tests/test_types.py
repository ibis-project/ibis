import pandas as pd

from ibis.clickhouse import api


def test_column_types(alltypes):
    df = alltypes.execute()
    assert df.tinyint_col.dtype.name == 'int8'
    assert df.smallint_col.dtype.name == 'int16'
    assert df.int_col.dtype.name == 'int32'
    assert df.bigint_col.dtype.name == 'int64'
    assert df.float_col.dtype.name == 'float32'
    assert df.double_col.dtype.name == 'float64'
    assert pd.core.common.is_datetime64_dtype(df.timestamp_col.dtype)


# def test_char_varchar_types(con):
#     sql = """\
# SELECT CAST(string_col AS varchar(20)) AS varchar_col,
#        CAST(string_col AS CHAR(5)) AS char_col
# FROM ibis_testing.`functional_alltypes`"""

#     t = con.sql(sql)
#     assert isinstance(t.varchar_col, api.StringColumn)
#     assert isinstance(t.char_col, api.StringColumn)
