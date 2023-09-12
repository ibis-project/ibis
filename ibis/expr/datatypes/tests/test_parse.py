from __future__ import annotations

import string

import hypothesis as h
import hypothesis.strategies as st
import parsy
import pytest

import ibis.expr.datatypes as dt
import ibis.tests.strategies as its
from ibis.common.annotations import ValidationError


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("boolean", dt.boolean),
        ("int8", dt.int8),
        ("int16", dt.int16),
        ("int32", dt.int32),
        ("int64", dt.int64),
        ("int", dt.int64),
        ("uint8", dt.uint8),
        ("uint16", dt.uint16),
        ("uint32", dt.uint32),
        ("uint64", dt.uint64),
        ("float16", dt.float16),
        ("float32", dt.float32),
        ("float64", dt.float64),
        ("float", dt.float64),
        ("string", dt.string),
        ("binary", dt.binary),
        ("date", dt.date),
        ("time", dt.time),
        ("timestamp", dt.timestamp),
        ("point", dt.point),
        ("linestring", dt.linestring),
        ("polygon", dt.polygon),
        ("multilinestring", dt.multilinestring),
        ("multipoint", dt.multipoint),
        ("multipolygon", dt.multipolygon),
    ],
)
def test_primitive_from_string(spec, expected):
    assert dt.dtype(spec) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ["decimal", dt.Decimal(None, None)],
        ["decimal(10, 3)", dt.Decimal(10, 3)],
        ["bignumeric", dt.Decimal(76, 38)],
        ["bigdecimal", dt.Decimal(76, 38)],
        ["bignumeric(1, 1)", dt.Decimal(1, 1)],
        ["bigdecimal(1, 1)", dt.Decimal(1, 1)],
    ],
)
def test_parse_decimal(spec, expected):
    assert dt.dtype(spec) == expected


@pytest.mark.parametrize(
    "case",
    [
        "decimal(",
        "decimal()",
        "decimal(3)",
        "decimal(,)",
        "decimal(3,)",
        "decimal(3,",
    ],
)
def test_parse_decimal_failure(case):
    with pytest.raises(parsy.ParseError):
        dt.dtype(case)


@pytest.mark.parametrize("spec", ["varchar", "varchar(10)", "char", "char(10)"])
def test_parse_char_varchar(spec):
    assert dt.dtype(spec) == dt.string


@pytest.mark.parametrize(
    "spec", ["varchar(", "varchar)", "varchar()", "char(", "char)", "char()"]
)
def test_parse_char_varchar_invalid(spec):
    with pytest.raises(parsy.ParseError):
        dt.dtype(spec)


def test_parse_array_token_error():
    with pytest.raises(parsy.ParseError):
        dt.dtype("array<string>>")


def test_parse_struct():
    orders = """array<struct<
                    oid: int64,
                    status: string,
                    totalprice: decimal(12, 2),
                    order_date: string,
                    items: array<struct<
                        iid: int64,
                        name: string,
                        price: decimal(12, 2),
                        discount_perc: decimal(12, 2),
                        shipdate: string,
                        : bool
                    >>
                >>"""
    expected = dt.Array(
        dt.Struct.from_tuples(
            [
                ("oid", dt.int64),
                ("status", dt.string),
                ("totalprice", dt.Decimal(12, 2)),
                ("order_date", dt.string),
                (
                    "items",
                    dt.Array(
                        dt.Struct.from_tuples(
                            [
                                ("iid", dt.int64),
                                ("name", dt.string),
                                ("price", dt.Decimal(12, 2)),
                                ("discount_perc", dt.Decimal(12, 2)),
                                ("shipdate", dt.string),
                                ("", dt.boolean),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    assert dt.dtype(orders) == expected


def test_struct_with_string_types():
    result = dt.Struct.from_tuples(
        [
            ("a", "map<double, string>"),
            ("b", "array<map<string, array<int32>>>"),
            ("c", "array<string>"),
            ("d", "int8"),
        ]
    )

    assert result == dt.Struct.from_tuples(
        [
            ("a", dt.Map(dt.double, dt.string)),
            ("b", dt.Array(dt.Map(dt.string, dt.Array(dt.int32)))),
            ("c", dt.Array(dt.string)),
            ("d", dt.int8),
        ]
    )


def test_array_with_string_value_types():
    assert dt.Array("int32") == dt.Array(dt.int32)
    assert dt.Array(dt.Array("array<map<string, double>>")) == (
        dt.Array(dt.Array(dt.Array(dt.Map(dt.string, dt.double))))
    )


def test_map_with_string_value_types():
    assert dt.Map("int32", "double") == dt.Map(dt.int32, dt.double)
    assert dt.Map("int32", "array<double>") == dt.Map(dt.int32, dt.Array(dt.double))


def test_parse_empty_map_failure():
    with pytest.raises(parsy.ParseError):
        dt.dtype("map<>")


def test_parse_map_allow_non_primitive_keys():
    assert dt.dtype("map<array<string>, double>") == dt.Map(
        dt.Array(dt.string), dt.double
    )


def test_parse_timestamp_with_timezone_single_quote():
    t = dt.dtype("timestamp('US/Eastern')")
    assert isinstance(t, dt.Timestamp)
    assert t.timezone == "US/Eastern"


def test_parse_timestamp_with_timezone_double_quote():
    t = dt.dtype("timestamp('US/Eastern')")
    assert isinstance(t, dt.Timestamp)
    assert t.timezone == "US/Eastern"


def test_parse_timestamp_with_timezone_invalid_timezone():
    ts = dt.dtype("timestamp('US/Ea')")
    assert str(ts) == "timestamp('US/Ea')"


@pytest.mark.parametrize("scale", range(10))
@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
def test_parse_timestamp_with_scale(scale, tz):
    expected = dt.Timestamp(timezone=tz, scale=scale)
    typestring = f"timestamp({tz!r}, {scale:d})"
    assert dt.parse(typestring) == expected
    assert str(expected) == typestring


@pytest.mark.parametrize("scale", range(10))
def test_parse_timestamp_with_scale_no_tz(scale):
    assert dt.parse(f"timestamp({scale:d})") == dt.Timestamp(scale=scale)


@pytest.mark.parametrize(
    "unit",
    [
        "Y",
        "Q",
        "M",
        "W",
        "D",  # date units
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",  # time units
    ],
)
def test_parse_interval(unit):
    definition = f"interval('{unit}')"
    assert dt.Interval(unit) == dt.dtype(definition)


@pytest.mark.parametrize("unit", ["X", "unsupported"])
def test_parse_interval_with_invalid_unit(unit):
    definition = f"interval('{unit}')"
    with pytest.raises(ValidationError):
        dt.dtype(definition)


@pytest.mark.parametrize(
    "case",
    [
        "timestamp(US/Ea)",
        "timestamp('US/Eastern\")",
        "timestamp(\"US/Eastern')",
        "interval(Y)",
        "interval('Y\")",
        "interval(\"Y')",
    ],
)
def test_parse_temporal_with_invalid_string_argument(case):
    with pytest.raises(parsy.ParseError):
        dt.dtype(case)


def test_parse_time():
    assert dt.dtype("time").equals(dt.time)


def test_parse_null():
    assert dt.parse("null") == dt.null


# corresponds to its.all_dtypes() but without:
# - geospacial types, the string representation is different from what the parser expects
# - struct types, the generated struct field names contain special characters

field_names = st.text(
    alphabet=st.characters(
        whitelist_characters=string.ascii_letters + string.digits,
        whitelist_categories=(),
    )
)

roundtrippable_dtypes = st.deferred(
    lambda: (
        its.primitive_dtypes()
        | its.string_like_dtypes()
        | its.temporal_dtypes()
        | its.interval_dtype()
        | its.variadic_dtypes()
        | its.struct_dtypes(names=field_names)
        | its.array_dtypes(roundtrippable_dtypes)
        | its.map_dtypes(roundtrippable_dtypes, roundtrippable_dtypes)
    )
)


@h.given(roundtrippable_dtypes)
def test_parse_dtype_roundtrip(dtype):
    assert dt.dtype(str(dtype)) == dtype
