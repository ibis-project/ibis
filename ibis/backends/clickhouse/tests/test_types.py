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
    sql_types = ["toFixedString('foo', 8) AS fixedstring_col"]
    if con.version.base_version >= '1.1.54337':
        sql_types.append(
            "toDateTime('2018-07-02 00:00:00', 'UTC') AS datetime_col"
        )
    sql = 'SELECT {}'.format(', '.join(sql_types))
    df = con.sql(sql).execute()
    assert df.fixedstring_col.dtype.name == 'object'
    if con.version.base_version >= '1.1.54337':
        assert df.datetime_col.dtype.name == 'datetime64[ns]'
