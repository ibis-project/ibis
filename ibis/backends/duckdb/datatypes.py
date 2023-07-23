from __future__ import annotations

import duckdb_engine.datatypes as ducktypes
import parsy
import sqlalchemy.dialects.postgresql as psql
import toolz

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.common.parsing import (
    COMMA,
    FIELD,
    LBRACKET,
    LPAREN,
    PRECISION,
    RAW_STRING,
    RBRACKET,
    RPAREN,
    SCALE,
    spaceless,
    spaceless_string,
)


def parse(text: str, default_decimal_parameters=(18, 3)) -> dt.DataType:
    """Parse a DuckDB type into an ibis data type."""
    primitive = (
        spaceless_string("interval").result(dt.Interval('us'))
        | spaceless_string("hugeint", "int128").result(dt.Decimal(38, 0))
        | spaceless_string("bigint", "int8", "long").result(dt.int64)
        | spaceless_string("boolean", "bool", "logical").result(dt.boolean)
        | spaceless_string("blob", "bytea", "binary", "varbinary").result(dt.binary)
        | spaceless_string("double", "float8").result(dt.float64)
        | spaceless_string("real", "float4", "float").result(dt.float32)
        | spaceless_string("smallint", "int2", "short").result(dt.int16)
        | spaceless_string(
            "timestamp with time zone", "timestamp_tz", "datetime"
        ).result(dt.Timestamp(timezone="UTC"))
        | spaceless_string("timestamp_sec", "timestamp_s").result(
            dt.Timestamp(timezone="UTC", scale=0)
        )
        | spaceless_string("timestamp_ms").result(dt.Timestamp(timezone="UTC", scale=3))
        | spaceless_string("timestamp_us").result(dt.Timestamp(timezone="UTC", scale=6))
        | spaceless_string("timestamp_ns").result(dt.Timestamp(timezone="UTC", scale=9))
        | spaceless_string("timestamp").result(dt.Timestamp(timezone="UTC"))
        | spaceless_string("date").result(dt.date)
        | spaceless_string("time").result(dt.time)
        | spaceless_string("tinyint", "int1").result(dt.int8)
        | spaceless_string("integer", "int4", "int", "signed").result(dt.int32)
        | spaceless_string("ubigint").result(dt.uint64)
        | spaceless_string("usmallint").result(dt.uint16)
        | spaceless_string("uinteger").result(dt.uint32)
        | spaceless_string("utinyint").result(dt.uint8)
        | spaceless_string("uuid").result(dt.uuid)
        | spaceless_string("varchar", "char", "bpchar", "text", "string").result(
            dt.string
        )
        | spaceless_string("json").result(dt.json)
        | spaceless_string("null").result(dt.null)
    )

    decimal = spaceless_string("decimal", "numeric").then(
        parsy.seq(LPAREN.then(PRECISION), COMMA.then(SCALE).skip(RPAREN))
        .optional(default_decimal_parameters)
        .combine(dt.Decimal)
    )

    brackets = spaceless(LBRACKET).then(spaceless(RBRACKET))

    ty = parsy.forward_declaration()
    non_pg_array_type = parsy.forward_declaration()

    pg_array = parsy.seq(non_pg_array_type, brackets.at_least(1).map(len)).combine(
        lambda value_type, n: toolz.nth(n, toolz.iterate(dt.Array, value_type))
    )

    map = (
        spaceless_string("map")
        .then(LPAREN)
        .then(parsy.seq(ty, COMMA.then(ty)).combine(dt.Map))
        .skip(RPAREN)
    )

    field = spaceless(parsy.alt(FIELD, RAW_STRING))

    struct = (
        spaceless_string("struct")
        .then(LPAREN)
        .then(parsy.seq(field, ty).sep_by(COMMA).map(dt.Struct.from_tuples))
        .skip(RPAREN)
    )

    non_pg_array_type.become(primitive | decimal | map | struct)
    ty.become(pg_array | non_pg_array_type)
    return ty.parse(text)


_from_duckdb_types = {
    psql.BYTEA: dt.Binary,
    psql.UUID: dt.UUID,
    ducktypes.TinyInteger: dt.Int8,
    ducktypes.SmallInteger: dt.Int16,
    ducktypes.Integer: dt.Int32,
    ducktypes.BigInteger: dt.Int64,
    ducktypes.HugeInteger: dt.Decimal(38, 0),
    ducktypes.UInt8: dt.UInt8,
    ducktypes.UTinyInteger: dt.UInt8,
    ducktypes.UInt16: dt.UInt16,
    ducktypes.USmallInteger: dt.UInt16,
    ducktypes.UInt32: dt.UInt32,
    ducktypes.UInteger: dt.UInt32,
    ducktypes.UInt64: dt.UInt64,
    ducktypes.UBigInteger: dt.UInt64,
}

_to_duckdb_types = {
    dt.UUID: psql.UUID,
    dt.Int8: ducktypes.TinyInteger,
    dt.Int16: ducktypes.SmallInteger,
    dt.Int32: ducktypes.Integer,
    dt.Int64: ducktypes.BigInteger,
    dt.UInt8: ducktypes.UTinyInteger,
    dt.UInt16: ducktypes.USmallInteger,
    dt.UInt32: ducktypes.UInteger,
    dt.UInt64: ducktypes.UBigInteger,
}


class DuckDBType(AlchemyType):
    dialect = "duckdb"

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if dtype := _from_duckdb_types.get(type(typ)):
            return dtype(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        if typ := _to_duckdb_types.get(type(dtype)):
            return typ
        else:
            return super().from_ibis(dtype)
