from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.snowflake.tests.conftest import _get_url
from ibis.backends.sql.datatypes import SnowflakeType
from ibis.util import gen_name

dtypes = [
    ("FIXED", dt.int64),
    ("REAL", dt.float64),
    ("TEXT", dt.string),
    ("DATE", dt.date),
    ("TIMESTAMP", dt.Timestamp(scale=9)),
    ("VARIANT", dt.json),
    ("TIMESTAMP_LTZ", dt.Timestamp(timezone="UTC", scale=9)),
    ("TIMESTAMP_TZ", dt.Timestamp(timezone="UTC", scale=9)),
    ("TIMESTAMP_NTZ", dt.Timestamp(scale=9)),
    ("OBJECT", dt.Map(dt.string, dt.json)),
    ("ARRAY", dt.Array(dt.json)),
    ("BINARY", dt.binary),
    ("TIME", dt.time),
    ("BOOLEAN", dt.boolean),
]


@pytest.mark.parametrize(
    ("snowflake_type", "ibis_type"),
    [
        param(snowflake_type, ibis_type, id=snowflake_type)
        for snowflake_type, ibis_type in dtypes
    ],
)
def test_parse(snowflake_type, ibis_type):
    assert SnowflakeType.from_string(snowflake_type.upper()) == ibis_type


@pytest.fixture(scope="module")
def con():
    return ibis.connect(_get_url())


user_dtypes = [
    ("NUMBER", dt.int64),
    ("DECIMAL", dt.int64),
    ("NUMERIC", dt.int64),
    ("NUMBER(5)", dt.int64),
    ("DECIMAL(5, 2)", dt.Decimal(5, 2)),
    ("NUMERIC(21, 17)", dt.Decimal(21, 17)),
    ("INT", dt.int64),
    ("INTEGER", dt.int64),
    ("BIGINT", dt.int64),
    ("SMALLINT", dt.int64),
    ("TINYINT", dt.int64),
    ("BYTEINT", dt.int64),
    ("FLOAT", dt.float64),
    ("FLOAT4", dt.float64),
    ("FLOAT8", dt.float64),
    ("DOUBLE", dt.float64),
    ("DOUBLE PRECISION", dt.float64),
    ("REAL", dt.float64),
    ("VARCHAR", dt.string),
    ("CHAR", dt.string),
    ("CHARACTER", dt.string),
    ("STRING", dt.string),
    ("TEXT", dt.string),
    ("BINARY", dt.binary),
    ("VARBINARY", dt.binary),
    ("BOOLEAN", dt.boolean),
    ("DATE", dt.date),
    ("TIME", dt.time),
    ("VARIANT", dt.json),
    ("OBJECT", dt.Map(dt.string, dt.json)),
    ("ARRAY", dt.Array(dt.json)),
]


@pytest.mark.parametrize(
    ("snowflake_type", "ibis_type"),
    [
        param(snowflake_type, ibis_type, id=snowflake_type)
        for snowflake_type, ibis_type in user_dtypes
    ],
)
def test_extract_type_from_table_query(con, snowflake_type, ibis_type):
    name = gen_name("test_extract_type_from_table")
    query = f'CREATE TEMP TABLE "{name}" ("a" {snowflake_type})'
    con.raw_sql(query).close()
    expected_schema = ibis.schema(dict(a=ibis_type))

    t = con.sql(f'SELECT "a" FROM "{name}"')
    assert t.schema() == expected_schema


@pytest.mark.parametrize(
    ("snowflake_type", "ibis_type"),
    [
        param("DATETIME", dt.Timestamp(scale=9)),
        param("TIMESTAMP", dt.Timestamp(scale=9)),
        param("TIMESTAMP(3)", dt.Timestamp(scale=3)),
        param("TIMESTAMP_LTZ", dt.Timestamp(timezone="UTC", scale=9)),
        param("TIMESTAMP_LTZ(3)", dt.Timestamp(timezone="UTC", scale=3)),
        param("TIMESTAMP_NTZ", dt.Timestamp(scale=9)),
        param("TIMESTAMP_NTZ(3)", dt.Timestamp(scale=3)),
        param("TIMESTAMP_TZ", dt.Timestamp(timezone="UTC", scale=9)),
        param("TIMESTAMP_TZ(3)", dt.Timestamp(timezone="UTC", scale=3)),
    ],
)
def test_extract_timestamp_from_table(con, snowflake_type, ibis_type):
    name = gen_name("test_extract_type_from_table")
    query = f'CREATE TEMP TABLE "{name}" ("a" {snowflake_type})'
    con.raw_sql(query).close()

    expected_schema = ibis.schema(dict(a=ibis_type))

    t = con.table(name)
    assert t.schema() == expected_schema


def test_array_discovery(con):
    t = con.tables.ARRAY_TYPES
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.json),
            y=dt.Array(dt.json),
            z=dt.Array(dt.json),
            grouper=dt.string,
            scalar_column=dt.float64,
            multi_dim=dt.Array(dt.json),
        )
    )
    assert t.schema() == expected
