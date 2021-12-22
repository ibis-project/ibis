import pytest
from pkg_resources import parse_version

import ibis.expr.datatypes as dt
from ibis.backends.clickhouse.client import ClickhouseDataType


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
    if parse_version(con.version).base_version >= '1.1.54337':
        sql_types.append(
            "toDateTime('2018-07-02 00:00:00', 'UTC') AS datetime_col"
        )
    sql = 'SELECT {}'.format(', '.join(sql_types))
    df = con.sql(sql).execute()
    assert df.fixedstring_col.dtype.name == 'object'
    if parse_version(con.version).base_version >= '1.1.54337':
        assert df.datetime_col.dtype.name == 'datetime64[ns]'


@pytest.mark.parametrize(
    ('ch_type', 'ibis_type'),
    [
        ('Array(Int8)', dt.Array(dt.Int8(nullable=False))),
        ('Array(Int16)', dt.Array(dt.Int16(nullable=False))),
        ('Array(Int32)', dt.Array(dt.Int32(nullable=False))),
        ('Array(Int64)', dt.Array(dt.Int64(nullable=False))),
        ('Array(UInt8)', dt.Array(dt.UInt8(nullable=False))),
        ('Array(UInt16)', dt.Array(dt.UInt16(nullable=False))),
        ('Array(UInt32)', dt.Array(dt.UInt32(nullable=False))),
        ('Array(UInt64)', dt.Array(dt.UInt64(nullable=False))),
        ('Array(Float32)', dt.Array(dt.Float32(nullable=False))),
        ('Array(Float64)', dt.Array(dt.Float64(nullable=False))),
        ('Array(String)', dt.Array(dt.String(nullable=False))),
        ('Array(FixedString(32))', dt.Array(dt.String(nullable=False))),
        ('Array(Date)', dt.Array(dt.Date(nullable=False))),
        ('Array(DateTime)', dt.Array(dt.Timestamp(nullable=False))),
        ('Array(DateTime64)', dt.Array(dt.Timestamp(nullable=False))),
        ('Array(Nothing)', dt.Array(dt.Null(nullable=False))),
        ('Array(Null)', dt.Array(dt.Null(nullable=False))),
        ('Array(Array(Int8))', dt.Array(dt.Array(dt.Int8(nullable=False)))),
        (
            'Array(Array(Array(Int8)))',
            dt.Array(dt.Array(dt.Array(dt.Int8(nullable=False)))),
        ),
        (
            'Array(Array(Array(Array(Int8))))',
            dt.Array(dt.Array(dt.Array(dt.Array(dt.Int8(nullable=False))))),
        ),
    ],
)
def test_array_type(ch_type, ibis_type):
    assert ClickhouseDataType(ch_type).to_ibis() == ibis_type
