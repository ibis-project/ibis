import pytest

import ibis.expr.datatypes as dt
from ibis.backends.clickhouse.datatypes import parse

pytest.importorskip("clickhouse_driver")


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
    sql_types = [
        "toFixedString('foo', 8) AS fixedstring_col",
        "toDateTime('2018-07-02 00:00:00', 'UTC') AS datetime_col",
    ]
    df = con.sql(f"SELECT {', '.join(sql_types)}").execute()
    assert df.fixedstring_col.dtype.name == 'object'
    assert df.datetime_col.dtype.name == 'datetime64[ns, UTC]'


@pytest.mark.parametrize(
    ('ch_type', 'ibis_type'),
    [
        (
            "Enum8('' = 0, 'CDMA' = 1, 'GSM' = 2, 'LTE' = 3, 'NR' = 4)",
            dt.String(nullable=False),
        ),
        ('IPv4', dt.inet(nullable=False)),
        ('IPv6', dt.inet(nullable=False)),
        ('JSON', dt.json(nullable=False)),
        ("Object('json')", dt.json(nullable=False)),
        ('LowCardinality(String)', dt.String(nullable=False)),
        ('Array(Int8)', dt.Array(dt.Int8(nullable=False), nullable=False)),
        ('Array(Int16)', dt.Array(dt.Int16(nullable=False), nullable=False)),
        ('Array(Int32)', dt.Array(dt.Int32(nullable=False), nullable=False)),
        ('Array(Int64)', dt.Array(dt.Int64(nullable=False), nullable=False)),
        ('Array(UInt8)', dt.Array(dt.UInt8(nullable=False), nullable=False)),
        ('Array(UInt16)', dt.Array(dt.UInt16(nullable=False), nullable=False)),
        ('Array(UInt32)', dt.Array(dt.UInt32(nullable=False), nullable=False)),
        ('Array(UInt64)', dt.Array(dt.UInt64(nullable=False), nullable=False)),
        (
            'Array(Float32)',
            dt.Array(dt.Float32(nullable=False), nullable=False),
        ),
        (
            'Array(Float64)',
            dt.Array(dt.Float64(nullable=False), nullable=False),
        ),
        ('Array(String)', dt.Array(dt.String(nullable=False), nullable=False)),
        (
            'Array(FixedString(32))',
            dt.Array(dt.String(nullable=False), nullable=False),
        ),
        ('Array(Date)', dt.Array(dt.Date(nullable=False), nullable=False)),
        (
            'Array(DateTime)',
            dt.Array(dt.Timestamp(nullable=False), nullable=False),
        ),
        (
            'Array(DateTime64)',
            dt.Array(dt.Timestamp(nullable=False), nullable=False),
        ),
        ('Array(Nothing)', dt.Array(dt.null, nullable=False)),
        ('Array(Null)', dt.Array(dt.null, nullable=False)),
        (
            'Array(Array(Int8))',
            dt.Array(
                dt.Array(dt.Int8(nullable=False), nullable=False),
                nullable=False,
            ),
        ),
        (
            'Array(Array(Array(Int8)))',
            dt.Array(
                dt.Array(
                    dt.Array(dt.Int8(nullable=False), nullable=False),
                    nullable=False,
                ),
                nullable=False,
            ),
        ),
        (
            'Array(Array(Array(Array(Int8))))',
            dt.Array(
                dt.Array(
                    dt.Array(
                        dt.Array(dt.Int8(nullable=False), nullable=False),
                        nullable=False,
                    ),
                    nullable=False,
                ),
                nullable=False,
            ),
        ),
        (
            "Map(Nullable(String), Nullable(UInt64))",
            dt.Map(dt.string, dt.uint64, nullable=False),
        ),
        ("Decimal(10, 3)", dt.Decimal(10, 3, nullable=False)),
        (
            "Tuple(a String, b Array(Nullable(Float64)))",
            dt.Struct.from_dict(
                dict(
                    a=dt.String(nullable=False),
                    b=dt.Array(dt.float64, nullable=False),
                ),
                nullable=False,
            ),
        ),
        (
            "Tuple(String, Array(Nullable(Float64)))",
            dt.Struct.from_dict(
                dict(
                    f0=dt.String(nullable=False),
                    f1=dt.Array(dt.float64, nullable=False),
                ),
                nullable=False,
            ),
        ),
        (
            "Tuple(a String, Array(Nullable(Float64)))",
            dt.Struct.from_dict(
                dict(
                    a=dt.String(nullable=False),
                    f1=dt.Array(dt.float64, nullable=False),
                ),
                nullable=False,
            ),
        ),
        (
            "Nested(a String, b Array(Nullable(Float64)))",
            dt.Struct.from_dict(
                dict(
                    a=dt.Array(dt.String(nullable=False), nullable=False),
                    b=dt.Array(
                        dt.Array(dt.float64, nullable=False), nullable=False
                    ),
                ),
                nullable=False,
            ),
        ),
    ],
)
def test_parse_type(ch_type, ibis_type):
    assert parse(ch_type) == ibis_type
