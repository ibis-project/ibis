from __future__ import annotations

import numpy as np
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.sql.datatypes import DuckDBType


@pytest.mark.parametrize(
    ("typ", "expected"),
    [
        param(typ, expected, id=typ.lower())
        for typ, expected in [
            ("BIGINT", dt.int64),
            ("BOOLEAN", dt.boolean),
            ("BLOB", dt.binary),
            ("DATE", dt.date),
            ("DOUBLE", dt.float64),
            ("DECIMAL(10, 3)", dt.Decimal(10, 3)),
            ("INTEGER", dt.int32),
            ("INTERVAL", dt.Interval("us")),
            ("FLOAT", dt.float32),
            ("SMALLINT", dt.int16),
            ("TIME", dt.time),
            ("TIME WITH TIME ZONE", dt.time),
            ("TIMESTAMP", dt.Timestamp(scale=6)),
            ("TIMESTAMP WITH TIME ZONE", dt.Timestamp(scale=6, timezone="UTC")),
            ("TINYINT", dt.int8),
            ("UBIGINT", dt.uint64),
            ("UINTEGER", dt.uint32),
            ("USMALLINT", dt.uint16),
            ("UTINYINT", dt.uint8),
            ("UUID", dt.uuid),
            ("VARCHAR", dt.string),
            ("INTEGER[]", dt.Array(dt.int32)),
            ("MAP(VARCHAR, BIGINT)", dt.Map(dt.string, dt.int64)),
            (
                "STRUCT(a INTEGER, b VARCHAR, c MAP(VARCHAR, DOUBLE[])[])",
                dt.Struct(
                    dict(
                        a=dt.int32,
                        b=dt.string,
                        c=dt.Array(dt.Map(dt.string, dt.Array(dt.float64))),
                    )
                ),
            ),
            ("INTEGER[][]", dt.Array(dt.Array(dt.int32))),
            ("JSON", dt.json),
            ("HUGEINT", dt.Decimal(38, 0)),
            ("TIMESTAMP_S", dt.Timestamp(scale=0)),
            ("TIMESTAMP_MS", dt.Timestamp(scale=3)),
            ("TIMESTAMP_NS", dt.Timestamp(scale=9)),
        ]
    ],
)
def test_parser(typ, expected):
    ty = DuckDBType.from_string(typ)
    assert ty == expected


@pytest.mark.parametrize("uint_type", ["uint8", "uint16", "uint32", "uint64"])
def test_cast_uints(uint_type, snapshot):
    t = ibis.table(dict(a="int8"), name="t")
    snapshot.assert_match(
        str(ibis.to_sql(t.a.cast(uint_type), dialect="duckdb")), "out.sql"
    )


def test_null_dtype():
    con = ibis.connect("duckdb://:memory:")

    t = ibis.memtable({"a": [None, None]})
    assert t.schema() == ibis.schema(dict(a="null"))

    df = con.execute(t)
    assert df.a.tolist() == [None, None]


def test_parse_quoted_struct_field():
    assert DuckDBType.from_string('STRUCT("a" INTEGER, "a b c" INTEGER)') == dt.Struct(
        {"a": dt.int32, "a b c": dt.int32}
    )


def test_read_uint8_from_parquet(tmp_path):
    con = ibis.duckdb.connect()

    # There is an incorrect mapping in duckdb-engine from UInteger -> UInt8
    # In order to get something that reads as a UInt8, we cast to UInt32 (UInteger)
    t = ibis.memtable({"a": np.array([1, 2, 3, 4], dtype="uint32")})
    assert t.a.type() == dt.uint32

    parqpath = tmp_path / "uint.parquet"

    con.to_parquet(t, parqpath)

    # If this doesn't fail, then things are working
    t2 = con.read_parquet(parqpath)

    assert t2.schema() == t.schema()


@pytest.mark.parametrize("typ", ["float32", "float64"])
def test_cast_to_floating_point_type(con, snapshot, typ):
    expected = 1.0
    value = ibis.literal(str(expected))
    expr = value.cast(typ)

    result = con.execute(expr)
    assert result == expected

    sql = str(ibis.to_sql(expr, dialect="duckdb"))
    snapshot.assert_match(sql, "out.sql")
