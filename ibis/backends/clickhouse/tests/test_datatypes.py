from __future__ import annotations

import hypothesis as h
import hypothesis.strategies as st
import pytest
import sqlglot as sg
from packaging.version import parse as vparse
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.tests.strategies as its
from ibis.backends.sql.datatypes import ClickHouseType

pytest.importorskip("clickhouse_connect")


def test_column_types(alltypes):
    df = alltypes.limit(1).execute()
    assert df.tinyint_col.dtype.name == "int8"
    assert df.smallint_col.dtype.name == "int16"
    assert df.int_col.dtype.name == "int32"
    assert df.bigint_col.dtype.name == "int64"
    assert df.float_col.dtype.name == "float32"
    assert df.double_col.dtype.name == "float64"
    assert df.timestamp_col.dtype.name == "datetime64[ns]"


def test_columns_types_with_additional_argument(con):
    sql_types = [
        "toFixedString('foo', 8) AS fixedstring_col",
        "toDateTime('2018-07-02 00:00:00', 'UTC') AS datetime_col",
        "toDateTime64('2018-07-02 00:00:00', 9, 'UTC') AS datetime_ns_col",
    ]
    df = con.sql(f"SELECT {', '.join(sql_types)}").execute()
    assert df.fixedstring_col.dtype.name == "object"
    assert df.datetime_col.dtype.name in ("datetime64[ns, UTC]", "datetime64[s, UTC]")
    assert df.datetime_ns_col.dtype.name == "datetime64[ns, UTC]"


def test_array_discovery_clickhouse(con):
    t = con.tables.array_types
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.int64, nullable=False),
            y=dt.Array(dt.string, nullable=False),
            z=dt.Array(dt.float64, nullable=False),
            grouper=dt.string,
            scalar_column=dt.float64,
            multi_dim=dt.Array(dt.Array(dt.int64, nullable=False), nullable=False),
        )
    )
    assert t.schema() == expected


@pytest.mark.parametrize(
    ("ch_type", "ibis_type"),
    [
        param(
            "Enum8('' = 0, 'CDMA' = 1, 'GSM' = 2, 'LTE' = 3, 'NR' = 4)",
            dt.String(nullable=False),
            id="enum",
        ),
        param("IPv4", dt.inet(nullable=False), id="ipv4"),
        param("IPv6", dt.inet(nullable=False), id="ipv6"),
        param("JSON", dt.json(nullable=False), id="json"),
        param("Object('json')", dt.json(nullable=False), id="object_json"),
        param(
            "LowCardinality(String)", dt.String(nullable=False), id="low_card_string"
        ),
        param(
            "Array(Int8)",
            dt.Array(dt.Int8(nullable=False), nullable=False),
            id="array_int8",
        ),
        param(
            "Array(Int16)",
            dt.Array(dt.Int16(nullable=False), nullable=False),
            id="array_int16",
        ),
        param(
            "Array(Int32)",
            dt.Array(dt.Int32(nullable=False), nullable=False),
            id="array_int32",
        ),
        param(
            "Array(Int64)",
            dt.Array(dt.Int64(nullable=False), nullable=False),
            id="array_int64",
        ),
        param(
            "Array(UInt8)",
            dt.Array(dt.UInt8(nullable=False), nullable=False),
            id="array_uint8",
        ),
        param(
            "Array(UInt16)",
            dt.Array(dt.UInt16(nullable=False), nullable=False),
            id="array_uint16",
        ),
        param(
            "Array(UInt32)",
            dt.Array(dt.UInt32(nullable=False), nullable=False),
            id="array_uint32",
        ),
        param(
            "Array(UInt64)",
            dt.Array(dt.UInt64(nullable=False), nullable=False),
            id="array_uint64",
        ),
        param(
            "Array(Float32)",
            dt.Array(dt.Float32(nullable=False), nullable=False),
            id="array_float32",
        ),
        param(
            "Array(Float64)",
            dt.Array(dt.Float64(nullable=False), nullable=False),
            id="array_float64",
        ),
        param(
            "Array(String)",
            dt.Array(dt.String(nullable=False), nullable=False),
            id="array_string",
        ),
        param(
            "Array(FixedString(32))",
            dt.Array(dt.String(nullable=False), nullable=False),
            id="array_fixed_string",
        ),
        param(
            "Array(Date)",
            dt.Array(dt.Date(nullable=False), nullable=False),
            id="array_date",
        ),
        param(
            "Array(DateTime)",
            dt.Array(dt.Timestamp(scale=0, nullable=False), nullable=False),
            id="array_datetime",
        ),
        param(
            "Array(DateTime64(9))",
            dt.Array(dt.Timestamp(scale=9, nullable=False), nullable=False),
            id="array_datetime64",
        ),
        param("Array(Nothing)", dt.Array(dt.null, nullable=False), id="array_nothing"),
        param("Array(Null)", dt.Array(dt.null, nullable=False), id="array_null"),
        param(
            "Array(Array(Int8))",
            dt.Array(
                dt.Array(dt.Int8(nullable=False), nullable=False),
                nullable=False,
            ),
            id="double_array",
        ),
        param(
            "Array(Array(Array(Int8)))",
            dt.Array(
                dt.Array(
                    dt.Array(dt.Int8(nullable=False), nullable=False),
                    nullable=False,
                ),
                nullable=False,
            ),
            id="triple_array",
        ),
        param(
            "Array(Array(Array(Array(Int8))))",
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
            id="quad_array",
        ),
        param(
            "Map(Nullable(String), Nullable(UInt64))",
            dt.Map(dt.string, dt.uint64, nullable=False),
            id="map",
        ),
        param("Decimal(10, 3)", dt.Decimal(10, 3, nullable=False), id="decimal"),
        param(
            "Tuple(a String, b Array(Nullable(Float64)))",
            dt.Struct(
                dict(
                    a=dt.String(nullable=False),
                    b=dt.Array(dt.float64, nullable=False),
                ),
                nullable=False,
            ),
            marks=pytest.mark.xfail(
                vparse(sg.__version__) == vparse("24.0.0"),
                reason="struct parsing for clickhouse broken in sqlglot 24",
                raises=sg.ParseError,
            ),
            id="named_tuple",
        ),
        param(
            "Tuple(String, Array(Nullable(Float64)))",
            dt.Struct(
                dict(
                    f0=dt.String(nullable=False),
                    f1=dt.Array(dt.float64, nullable=False),
                ),
                nullable=False,
            ),
            marks=pytest.mark.xfail(
                vparse("24.0.0") <= vparse(sg.__version__) <= vparse("24.0.1"),
                reason="struct parsing for clickhouse broken in sqlglot 24",
                raises=sg.ParseError,
            ),
            id="unnamed_tuple",
        ),
        param(
            "Tuple(a String, Array(Nullable(Float64)))",
            dt.Struct(
                dict(
                    a=dt.String(nullable=False),
                    f1=dt.Array(dt.float64, nullable=False),
                ),
                nullable=False,
            ),
            marks=pytest.mark.xfail(
                vparse("24.0.0") <= vparse(sg.__version__) <= vparse("24.0.1"),
                reason="struct parsing for clickhouse broken in sqlglot 24",
                raises=sg.ParseError,
            ),
            id="partially_named",
        ),
        param(
            "Nested(a String, b Array(Nullable(Float64)))",
            dt.Struct(
                dict(
                    a=dt.Array(dt.String(nullable=False), nullable=False),
                    b=dt.Array(dt.Array(dt.float64, nullable=False), nullable=False),
                ),
                nullable=False,
            ),
            id="nested",
        ),
        param("Date32", dt.Date(nullable=False), id="date32"),
        param("DateTime", dt.Timestamp(scale=0, nullable=False), id="datetime"),
        param(
            "DateTime('Europe/Budapest')",
            dt.Timestamp(scale=0, timezone="Europe/Budapest", nullable=False),
            id="datetime_timezone",
        ),
        param(
            "DateTime64(0)", dt.Timestamp(scale=0, nullable=False), id="datetime64_zero"
        ),
        param(
            "DateTime64(1)", dt.Timestamp(scale=1, nullable=False), id="datetime64_one"
        ),
    ]
    + [
        param(
            f"DateTime64({scale}, '{tz}')",
            dt.Timestamp(scale=scale, timezone=tz, nullable=False),
            id=f"datetime64_{scale}_{tz}",
        )
        for scale in range(10)
        for tz in ("UTC", "America/New_York", "America/Chicago", "America/Los_Angeles")
    ],
)
def test_parse_type(ch_type, ibis_type):
    parsed_ibis_type = ClickHouseType.from_string(ch_type)
    assert parsed_ibis_type == ibis_type


false = st.just(False)

map_key_types = (
    its.string_dtype(nullable=false)
    | its.integer_dtypes(nullable=false)
    | its.date_dtype(nullable=false)
    | its.timestamp_dtype(scale=st.integers(0, 9), nullable=false)
)

roundtrippable_types = st.deferred(
    lambda: (
        its.null_dtype
        | its.boolean_dtype()
        | its.integer_dtypes()
        | st.just(dt.Float32())
        | st.just(dt.Float64())
        | its.decimal_dtypes()
        | its.string_dtype()
        | its.json_dtype()
        | its.inet_dtype()
        | its.uuid_dtype()
        | its.date_dtype()
        | its.time_dtype()
        | its.timestamp_dtype(scale=st.integers(0, 9))
        | its.array_dtypes(roundtrippable_types, nullable=false, length=st.none())
        | its.map_dtypes(map_key_types, roundtrippable_types, nullable=false)
    )
)


@h.given(roundtrippable_types)
def test_type_roundtrip(ibis_type):
    type_string = ClickHouseType.to_string(ibis_type)
    parsed_ibis_type = ClickHouseType.from_string(type_string)
    assert parsed_ibis_type == ibis_type


def test_arrays_nullable():
    sge_type = ClickHouseType.from_ibis(dt.Array("float"))
    assert sge_type.sql("clickhouse") == "Array(Nullable(Float64))"
