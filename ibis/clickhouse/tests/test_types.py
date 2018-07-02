import pytest


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
    assert df.timestamp_col.dtype.name == 'datetime64[ns]'


def test_columns_types_with_additional_argument(con):
    datetime_with_tz_sql = (
        "SELECT toDateTime('2018-07-02 00:00:00', 'UTC') AS datetime_with_tz")
    fixedstring_with_len_sql = (
        "SELECT toFixedString('foo', 8) as fixedstring_with_len")
    assert con.sql(datetime_with_tz_sql).execute()\
              .datetime_with_tz.dtype.name == 'datetime64[ns]'
    assert con.sql(fixedstring_with_len_sql).execute()\
              .fixedstring_with_len.dtype.name == 'object'
