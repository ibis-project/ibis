from __future__ import annotations

import parsy as p
import toolz

import ibis.expr.datatypes as dt
import ibis.util as util
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


def parse(text: str, default_decimal_parameters=(18, 3)) -> dt.DataType:
    """Parse a DuckDB type into an ibis data type."""
    primitive = (
        spaceless_string("interval").result(dt.Interval())
        | spaceless_string("bigint", "int8", "long").result(dt.int64)
        | spaceless_string("boolean", "bool", "logical").result(dt.boolean)
        | spaceless_string(
            "blob",
            "bytea",
            "binary",
            "varbinary",
        ).result(dt.binary)
        | spaceless_string("double", "float8").result(dt.float64)
        | spaceless_string("real", "float4", "float").result(dt.float32)
        | spaceless_string("smallint", "int2", "short").result(dt.int16)
        | spaceless_string("timestamp", "datetime").result(
            dt.Timestamp(timezone="UTC")
        )
        | spaceless_string("date").result(dt.date)
        | spaceless_string("time").result(dt.time)
        | spaceless_string("tinyint", "int1").result(dt.int8)
        | spaceless_string("integer", "int4", "int", "signed").result(dt.int32)
        | spaceless_string("ubigint").result(dt.uint64)
        | spaceless_string("usmallint").result(dt.uint16)
        | spaceless_string("uinteger").result(dt.uint32)
        | spaceless_string("utinyint").result(dt.uint8)
        | spaceless_string("uuid").result(dt.uuid)
        | spaceless_string(
            "varchar",
            "char",
            "bpchar",
            "text",
            "string",
        ).result(dt.string)
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
        return dt.Decimal(*prec_scale)

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
        return dt.Array(value_type)

    @p.generate
    def brackets():
        yield spaceless(LBRACKET)
        yield spaceless(RBRACKET)

    @p.generate
    def pg_array():
        value_type = yield non_pg_array_type
        n = len((yield brackets.at_least(1)))
        return toolz.nth(n, toolz.iterate(dt.Array, value_type))

    @p.generate
    def map():
        yield spaceless_string("map")
        yield LANGLE
        key_type = yield primitive
        yield COMMA
        value_type = yield ty
        yield RANGLE
        return dt.Map(key_type, value_type)

    field = spaceless(FIELD)

    @p.generate
    def struct():
        yield spaceless_string("struct")
        yield LPAREN
        field_names_types = yield (
            p.seq(field, ty)
            .combine(lambda field, ty: (field, ty))
            .sep_by(COMMA)
        )
        yield RPAREN
        return dt.Struct.from_tuples(field_names_types)

    non_pg_array_type = primitive | decimal | list_array | map | struct
    ty = pg_array | non_pg_array_type
    return ty.parse(text)


@util.deprecated(
    instead=f"use {parse.__module__}.{parse.__name__}",
    version="4.0",
)
def parse_type(*args, **kwargs):
    return parse(*args, **kwargs)
