from __future__ import annotations

import parsy as p
import toolz

from ibis import util
from ibis.common.parsing import (
    COMMA,
    FIELD,
    LANGLE,
    LBRACKET,
    LPAREN,
    PRECISION,
    RANGLE,
    RBRACKET,
    RPAREN,
    SCALE,
    spaceless,
    spaceless_string,
)
from ibis.expr.datatypes import (
    Array,
    DataType,
    Decimal,
    Interval,
    Map,
    Struct,
    Timestamp,
    binary,
    boolean,
    date,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    json,
    string,
    time,
    uint8,
    uint16,
    uint32,
    uint64,
    uuid,
)


def parse(text: str, default_decimal_parameters=(18, 3)) -> DataType:
    """Parse a DuckDB type into an ibis data type."""
    primitive = (
        spaceless_string("interval").result(Interval())
        | spaceless_string("bigint", "int8", "long").result(int64)
        | spaceless_string("boolean", "bool", "logical").result(boolean)
        | spaceless_string(
            "blob",
            "bytea",
            "binary",
            "varbinary",
        ).result(binary)
        | spaceless_string("double", "float8").result(float64)
        | spaceless_string("real", "float4", "float").result(float32)
        | spaceless_string("smallint", "int2", "short").result(int16)
        | spaceless_string(
            "timestamp with time zone",
            "timestamp_tz",
            "timestamp_sec",
            "timestamp_ms",
            "timestamp_ns",
            "timestamp",
            "datetime",
        ).result(Timestamp(timezone="UTC"))
        | spaceless_string("date").result(date)
        | spaceless_string("time").result(time)
        | spaceless_string("tinyint", "int1").result(int8)
        | spaceless_string("integer", "int4", "int", "signed").result(int32)
        | spaceless_string("ubigint").result(uint64)
        | spaceless_string("usmallint").result(uint16)
        | spaceless_string("uinteger").result(uint32)
        | spaceless_string("utinyint").result(uint8)
        | spaceless_string("uuid").result(uuid)
        | spaceless_string(
            "varchar",
            "char",
            "bpchar",
            "text",
            "string",
        ).result(string)
        | spaceless_string("json").result(json)
    )

    @p.generate
    def decimal():
        yield spaceless_string("decimal", "numeric")
        prec_scale = (
            yield LPAREN.then(
                p.seq(PRECISION.skip(COMMA), SCALE).combine(
                    lambda prec, scale: (prec, scale)
                )
            )
            .skip(RPAREN)
            .optional()
        ) or default_decimal_parameters
        return Decimal(*prec_scale)

    @p.generate
    def angle_type():
        yield LANGLE
        value_type = yield ty
        yield RANGLE
        return value_type

    @p.generate
    def list_array():
        yield spaceless_string("list")
        value_type = yield angle_type
        return Array(value_type)

    @p.generate
    def brackets():
        yield spaceless(LBRACKET)
        yield spaceless(RBRACKET)

    @p.generate
    def pg_array():
        value_type = yield non_pg_array_type
        n = len((yield brackets.at_least(1)))
        return toolz.nth(n, toolz.iterate(Array, value_type))

    @p.generate
    def map():
        yield spaceless_string("map")
        yield LANGLE
        key_type = yield primitive
        yield COMMA
        value_type = yield ty
        yield RANGLE
        return Map(key_type, value_type)

    field = spaceless(FIELD)

    @p.generate
    def struct():
        yield spaceless_string("struct")
        yield LPAREN
        field_names_types = yield (
            p.seq(field, ty).combine(lambda field, ty: (field, ty)).sep_by(COMMA)
        )
        yield RPAREN
        return Struct.from_tuples(field_names_types)

    non_pg_array_type = primitive | decimal | list_array | map | struct
    ty = pg_array | non_pg_array_type
    return ty.parse(text)


@util.deprecated(
    instead=f"use {parse.__module__}.{parse.__name__}", as_of="4.0", removed_in="5.0"
)
def parse_type(*args, **kwargs):
    return parse(*args, **kwargs)
