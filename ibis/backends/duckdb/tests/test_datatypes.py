from __future__ import annotations

from datetime import time

import numpy as np
import pyarrow as pa
import pytest
import sqlglot as sg
from packaging.version import parse as vparse
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
            ("ENUM('a', 'b')", dt.string),
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
            ("INTEGER[3]", dt.Array(dt.int32, length=3)),
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
    ]
    + [
        param(
            "TIME_NS",
            dt.time,
            marks=pytest.mark.xfail(
                vparse(sg.__version__) < vparse("28.1.0"),
                reason="TIME_NS was added in sqlglot 28.1.0",
            ),
            id="time_ns",
        ),
    ],
)
def test_parser(typ, expected):
    ty = DuckDBType.from_string(typ)
    assert ty == expected


def test_time_ns():
    import duckdb

    if vparse(duckdb.__version__) < vparse("1.5"):
        pytest.skip("DuckDB TIME_NS was added in DuckDB 1.5")

    con = ibis.duckdb.connect()

    # Null-free with sub-microsecond nanoseconds
    t1 = con.sql("SELECT '12:30:00.123456789'::TIME_NS AS t")
    assert t1.schema() == ibis.schema({"t": "time"})

    # PyArrow preserves exact nanosecond precision
    arrow_t1 = t1.to_pyarrow()
    assert arrow_t1.schema.field("t").type == pa.time64("ns")
    assert arrow_t1["t"][0].value == 45_000_123_456_789

    # Table with NULL
    t2 = con.sql("SELECT '12:30:00.123456789'::TIME_NS AS t UNION ALL SELECT NULL")
    assert t2.schema() == ibis.schema({"t": "time"})
    arrow_t2 = t2.to_pyarrow()
    assert arrow_t2.schema.field("t").type == pa.time64("ns")
    assert arrow_t2["t"][0].value == 45_000_123_456_789
    assert arrow_t2["t"][1].value is None

    # Polars preserves exact nanosecond precision when available
    pl = pytest.importorskip("polars")
    assert t1.to_polars()["t"].cast(pl.Int64).to_list() == [45_000_123_456_789]
    assert t2.to_polars()["t"].cast(pl.Int64).to_list() == [45_000_123_456_789, None]


@pytest.mark.filterwarnings("ignore::pandas.errors.Pandas4Warning")
def test_time_ns_execute():
    import duckdb

    if vparse(duckdb.__version__) < vparse("1.5"):
        pytest.skip("DuckDB TIME_NS was added in DuckDB 1.5")

    con = ibis.duckdb.connect()

    # Null-free: execute() truncates sub-microsecond digits to microseconds without error
    t1 = con.sql("SELECT '12:30:00.123456789'::TIME_NS AS t")
    assert t1.execute()["t"].tolist() == [time(12, 30, 0, 123456)]

    # With NULL: execute() also handles null rows alongside truncated times
    t2 = con.sql("SELECT '12:30:00.123456789'::TIME_NS AS t UNION ALL SELECT NULL")
    assert t2.execute()["t"].tolist() == [time(12, 30, 0, 123456), None]

    # Temporal field extractions
    assert t1.t.hour().execute().tolist() == [12]
    assert t1.t.minute().execute().tolist() == [30]
    assert t1.t.second().execute().tolist() == [0]
    assert t1.t.millisecond().execute().tolist() == [123]
    assert t1.t.microsecond().execute().tolist() == [123456]


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


def test_null_scalar(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    monkeypatch.setattr(ibis.options, "interactive", True)

    result = repr(ibis.literal(None, type="int"))
    expected = """\
┌──────┐
│ NULL │
└──────┘"""
    assert result == expected
