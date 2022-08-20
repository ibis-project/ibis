import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt

MYSQL_TYPES = [
    ("tinyint", dt.int8),
    ("int1", dt.int8),
    ("boolean", dt.int8),
    ("smallint", dt.int16),
    ("int2", dt.int16),
    ("mediumint", dt.int32),
    ("int3", dt.int32),
    ("int", dt.int32),
    ("int4", dt.int32),
    ("integer", dt.int32),
    ("bigint", dt.int64),
    ("decimal", dt.Decimal(10, 0)),
    ("decimal(5, 2)", dt.Decimal(5, 2)),
    ("dec", dt.Decimal(10, 0)),
    ("numeric", dt.Decimal(10, 0)),
    ("fixed", dt.Decimal(10, 0)),
    ("float", dt.float32),
    ("double", dt.float64),
    ("timestamp", dt.Timestamp("UTC")),
    ("date", dt.date),
    ("time", dt.time),
    ("datetime", dt.timestamp),
    ("year", dt.int16),
    ("char(32)", dt.string),
    ("char byte", dt.binary),
    ("varchar(42)", dt.string),
    ("mediumtext", dt.string),
    ("text", dt.string),
    ("binary(42)", dt.binary),
    ("varbinary(42)", dt.binary),
    ("bit(1)", dt.int8),
    ("bit(9)", dt.int16),
    ("bit(17)", dt.int32),
    ("bit(33)", dt.int64),
    # mariadb doesn't have a distinct json type
    ("json", dt.string),
    ("enum('small', 'medium', 'large')", dt.string),
    ("inet6", dt.string),
    ("set('a', 'b', 'c', 'd')", dt.Set(dt.string)),
    ("mediumblob", dt.binary),
    ("blob", dt.binary),
    ("uuid", dt.string),
]


@pytest.mark.parametrize(
    ("mysql_type", "expected_type"),
    [
        param(mysql_type, ibis_type, id=mysql_type)
        for mysql_type, ibis_type in MYSQL_TYPES
    ],
)
def test_get_schema_from_query(con, mysql_type, expected_type):
    raw_name = ibis.util.guid()
    name = con.con.dialect.identifier_preparer.quote_identifier(raw_name)
    # temporary tables get cleaned up by the db when the session ends, so we
    # don't need to explicitly drop the table
    con.raw_sql(f"CREATE TEMPORARY TABLE {name} (x {mysql_type})")
    expected_schema = ibis.schema(dict(x=expected_type))
    result_schema = con._get_schema_using_query(f"SELECT * FROM {name}")
    assert result_schema == expected_schema


@pytest.mark.parametrize(
    "coltype",
    ["TINYBLOB", "MEDIUMBLOB", "BLOB", "LONGBLOB"],
)
def test_blob_type(con, coltype):
    tmp = f"tmp_{ibis.util.guid()}"
    con.raw_sql(f"CREATE TEMPORARY TABLE {tmp} (a {coltype})")
    t = con.table(tmp)
    assert t.schema() == ibis.schema({"a": dt.binary})
