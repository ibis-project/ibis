import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt

DB_TYPES = [
    # Exact numbers
    ('BIGINT', dt.int64),
    ('BIT', dt.int8),
    ('DECIMAL', dt.Decimal(precision=18, scale=0)),
    ('DECIMAL(5, 2)', dt.Decimal(precision=5, scale=2)),
    ('INT', dt.int32),
    ('MONEY', dt.int64),
    ('NUMERIC', dt.Decimal(18, 0)),
    ('NUMERIC(10,5)', dt.Decimal(10, 5)),
    ('NUMERIC(14,3)', dt.Decimal(14, 3)),
    ('SMALLINT', dt.int16),
    ('SMALLMONEY', dt.int32),
    ('TINYINT', dt.int8),
    # Approximate numerics
    ('REAL', dt.float32),
    ('FLOAT', dt.float64),
    ('FLOAT(3)', dt.float32),
    ('FLOAT(25)', dt.float64),
    # Date and time
    ('DATE', dt.date),
    ('TIME', dt.time),
    ('DATETIME2', dt.timestamp),
    ('DATETIMEOFFSET', dt.timestamp),
    ('SMALLDATETIME', dt.timestamp),
    ('DATETIME', dt.timestamp),
    # Characters strings
    ('CHAR', dt.string),
    ('TEXT', dt.string),
    ('VARCHAR', dt.string),
    # Unicode character strings
    ('NCHAR', dt.string),
    ('NTEXT', dt.string),
    ('NVARCHAR', dt.string),
    # Binary strings
    ('BINARY', dt.binary),
    ('VARBINARY', dt.binary),
    ('IMAGE', dt.binary),
    # Other data types
    ('UNIQUEIDENTIFIER', dt.uuid),
    ('GEOMETRY', dt.geometry),
    ('GEOGRAPHY', dt.geography),
    ('TIMESTAMP', dt.binary),
]


@pytest.mark.parametrize(
    ("server_type", "expected_type"),
    [
        param(server_type, ibis_type, id=server_type)
        for server_type, ibis_type in DB_TYPES
    ],
)
def test_get_schema_from_query(con, server_type, expected_type):
    raw_name = f"tmp_{ibis.util.guid()}"
    name = con.con.dialect.identifier_preparer.quote_identifier(raw_name)
    try:
        con.raw_sql(f"CREATE TABLE {name} (x {server_type})")
        expected_schema = ibis.schema(dict(x=expected_type))
        result_schema = con._get_schema_using_query(f"SELECT * FROM {name}")
        assert result_schema == expected_schema
    finally:
        con.raw_sql(f"DROP TABLE IF EXISTS {name}")
