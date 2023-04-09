from __future__ import annotations

import parsy
import sqlalchemy as sa
import toolz
from sqlalchemy.dialects import postgresql

import ibis.backends.base.sql.alchemy.datatypes as sat
import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import to_sqla_type
from ibis.common.parsing import (
    COMMA,
    FIELD,
    LBRACKET,
    LPAREN,
    PRECISION,
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

    struct = (
        spaceless_string("struct")
        .then(LPAREN)
        .then(parsy.seq(spaceless(FIELD), ty).sep_by(COMMA).map(dt.Struct.from_tuples))
        .skip(RPAREN)
    )

    non_pg_array_type.become(primitive | decimal | map | struct)
    ty.become(pg_array | non_pg_array_type)
    return ty.parse(text)


try:
    from duckdb_engine import Dialect as DuckDBDialect
except ImportError:
    pass
else:

    @dt.dtype.register(DuckDBDialect, sat.UInt64)
    @dt.dtype.register(DuckDBDialect, sat.UInt32)
    @dt.dtype.register(DuckDBDialect, sat.UInt16)
    @dt.dtype.register(DuckDBDialect, sat.UInt8)
    def dtype_uint(_, satype, nullable=True):
        return getattr(dt, satype.__class__.__name__)(nullable=nullable)

    @dt.dtype.register(DuckDBDialect, sat.ArrayType)
    def _(dialect, satype, nullable=True):
        return dt.Array(dt.dtype(dialect, satype.value_type), nullable=nullable)

    @dt.dtype.register(DuckDBDialect, sat.MapType)
    def _(dialect, satype, nullable=True):
        return dt.Map(
            dt.dtype(dialect, satype.key_type),
            dt.dtype(dialect, satype.value_type),
            nullable=nullable,
        )

    @to_sqla_type.register(DuckDBDialect, dt.UUID)
    def sa_duckdb_uuid(*_):
        return postgresql.UUID()

    @to_sqla_type.register(DuckDBDialect, (dt.MACADDR, dt.INET))
    def sa_duckdb_macaddr(*_):
        return sa.TEXT()

    @to_sqla_type.register(DuckDBDialect, dt.Map)
    def sa_duckdb_map(dialect, itype):
        return sat.MapType(
            to_sqla_type(dialect, itype.key_type),
            to_sqla_type(dialect, itype.value_type),
        )

    @to_sqla_type.register(DuckDBDialect, dt.Array)
    def _(dialect, itype):
        return sat.ArrayType(to_sqla_type(dialect, itype.value_type))
