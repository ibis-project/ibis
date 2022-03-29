from __future__ import annotations

import re
from typing import TYPE_CHECKING

import parsy as p

if TYPE_CHECKING:
    from ibis.expr.datatypes import DataType

from ibis.expr.datatypes import (
    Array,
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
    string,
    time,
    uint8,
    uint16,
    uint32,
    uint64,
    uuid,
)

_SPACES = p.regex(r'\s*', re.MULTILINE)


def spaceless(parser):
    return _SPACES.then(parser).skip(_SPACES)


def spaceless_string(*strings: str):
    return spaceless(
        p.alt(*[p.string(s, transform=str.lower) for s in strings])
    )


def parse_type(text: str, default_decimal_parameters=(18, 3)) -> DataType:
    precision = scale = p.digit.at_least(1).concat().map(int)

    lparen = spaceless_string("(")
    rparen = spaceless_string(")")

    lbracket = spaceless_string("[")
    rbracket = spaceless_string("]")

    langle = spaceless_string("<")
    rangle = spaceless_string(">")

    comma = spaceless_string(",")
    colon = spaceless_string(":")

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
        | spaceless_string("timestamp", "datetime").result(
            Timestamp(timezone="UTC")
        )
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
    )

    @p.generate
    def decimal():
        yield spaceless_string("decimal", "numeric")
        prec_scale = (
            yield lparen.then(
                p.seq(precision.skip(comma), scale).combine(
                    lambda prec, scale: (prec, scale)
                )
            )
            .skip(rparen)
            .optional()
        ) or default_decimal_parameters
        return Decimal(*prec_scale)

    @p.generate
    def angle_type():
        yield langle
        value_type = yield ty
        yield rangle
        return value_type

    @p.generate
    def list_array():
        yield spaceless_string("list")
        value_type = yield angle_type
        return Array(value_type)

    @p.generate
    def pg_array():
        value_type = yield non_pg_array_type
        yield lbracket
        yield rbracket
        return Array(value_type)

    @p.generate
    def map():
        yield spaceless_string("map")
        yield langle
        key_type = yield primitive
        yield comma
        value_type = yield ty
        yield rangle
        return Map(key_type, value_type)

    field = spaceless(p.regex("[a-zA-Z_][a-zA-Z_0-9]*"))

    @p.generate
    def struct():
        yield spaceless_string("struct")
        yield langle
        field_names_types = yield (
            p.seq(field.skip(colon), ty)
            .combine(lambda field, ty: (field, ty))
            .sep_by(comma)
        )
        yield rangle
        return Struct.from_tuples(field_names_types)

    non_pg_array_type = primitive | decimal | list_array | map | struct
    ty = pg_array | non_pg_array_type
    return ty.parse(text)
