from __future__ import annotations

from functools import partial

import parsy
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from trino.sqlalchemy.datatype import DOUBLE, JSON, MAP, ROW, TIMESTAMP
from trino.sqlalchemy.dialect import TrinoDialect

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import to_sqla_type
from ibis.common.parsing import (
    COMMA,
    FIELD,
    LPAREN,
    PRECISION,
    RPAREN,
    SCALE,
    SINGLE_DIGIT,
    spaceless,
    spaceless_string,
)


def parse(text: str, default_decimal_parameters=(18, 3)) -> dt.DataType:
    """Parse a Trino type into an ibis data type."""

    timestamp = spaceless_string("timestamp").then(
        parsy.seq(
            scale=LPAREN.then(SINGLE_DIGIT.map(int)).skip(RPAREN).optional()
        ).combine_dict(partial(dt.Timestamp, timezone="UTC"))
    )

    primitive = (
        spaceless_string("interval").result(dt.Interval())
        | spaceless_string("bigint").result(dt.int64)
        | spaceless_string("boolean").result(dt.boolean)
        | spaceless_string("varbinary").result(dt.binary)
        | spaceless_string("double").result(dt.float64)
        | spaceless_string("real").result(dt.float32)
        | spaceless_string("smallint").result(dt.int16)
        | timestamp
        | spaceless_string("date").result(dt.date)
        | spaceless_string("time").result(dt.time)
        | spaceless_string("tinyint").result(dt.int8)
        | spaceless_string("integer").result(dt.int32)
        | spaceless_string("uuid").result(dt.uuid)
        | spaceless_string("varchar", "char").result(dt.string)
        | spaceless_string("json").result(dt.json)
        | spaceless_string("ipaddress").result(dt.inet)
    )

    decimal = spaceless_string("decimal", "numeric").then(
        parsy.seq(LPAREN.then(PRECISION).skip(COMMA), SCALE.skip(RPAREN))
        .optional(default_decimal_parameters)
        .combine(dt.Decimal)
    )

    ty = parsy.forward_declaration()

    array = spaceless_string("array").then(LPAREN).then(ty).skip(RPAREN).map(dt.Array)
    map = spaceless_string("map").then(
        parsy.seq(LPAREN.then(ty).skip(COMMA), ty.skip(RPAREN)).combine(dt.Map)
    )

    struct = (
        spaceless_string("row")
        .then(LPAREN)
        .then(parsy.seq(spaceless(FIELD), ty).sep_by(COMMA).map(dt.Struct.from_tuples))
        .skip(RPAREN)
    )

    ty.become(primitive | decimal | array | map | struct)
    return ty.parse(text)


@dt.dtype.register(TrinoDialect, DOUBLE)
def sa_trino_double(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


@dt.dtype.register(TrinoDialect, sa.REAL)
def sa_trino_real(_, satype, nullable=True):
    return dt.Float32(nullable=nullable)


@dt.dtype.register(TrinoDialect, sa.ARRAY)
def sa_trino_array(dialect, satype, nullable=True):
    value_dtype = dt.dtype(dialect, satype.item_type)
    return dt.Array(value_dtype, nullable=nullable)


@dt.dtype.register(TrinoDialect, ROW)
def sa_trino_row(dialect, satype, nullable=True):
    fields = ((name, dt.dtype(dialect, typ)) for name, typ in satype.attr_types)
    return dt.Struct.from_tuples(fields, nullable=nullable)


@dt.dtype.register(TrinoDialect, MAP)
def sa_trino_map(dialect, satype, nullable=True):
    return dt.Map(
        dt.dtype(dialect, satype.key_type),
        dt.dtype(dialect, satype.value_type),
        nullable=nullable,
    )


@dt.dtype.register(TrinoDialect, TIMESTAMP)
def sa_trino_timestamp(_, satype, nullable=True):
    return dt.Timestamp(
        timezone="UTC" if satype.timezone else None,
        scale=satype.precision,
        nullable=nullable,
    )


@dt.dtype.register(TrinoDialect, JSON)
def sa_trino_json(_, satype, nullable=True):
    return dt.JSON(nullable=nullable)


@to_sqla_type.register(TrinoDialect, dt.String)
def _string(_, itype):
    return sa.VARCHAR()


@to_sqla_type.register(TrinoDialect, dt.Struct)
def _struct(dialect, itype):
    return ROW(
        [(name, to_sqla_type(dialect, typ)) for name, typ in itype.fields.items()]
    )


@to_sqla_type.register(TrinoDialect, dt.Timestamp)
def _timestamp(_, itype):
    return TIMESTAMP(precision=itype.scale, timezone=bool(itype.timezone))


@compiles(TIMESTAMP)
def compiles_timestamp(typ, compiler, **kw):
    result = "TIMESTAMP"

    if (prec := typ.precision) is not None:
        result += f"({prec:d})"

    if typ.timezone:
        result += " WITH TIME ZONE"

    return result


@compiles(ROW)
def _compiles_row(element, compiler, **kw):
    # TODO: @compiles should live in the dialect
    quote = compiler.dialect.identifier_preparer.quote
    content = ", ".join(
        f"{quote(field)} {compiler.process(typ, **kw)}"
        for field, typ in element.attr_types
    )
    return f"ROW({content})"


@to_sqla_type.register(TrinoDialect, dt.Map)
def _map(dialect, itype):
    return MAP(
        to_sqla_type(dialect, itype.key_type), to_sqla_type(dialect, itype.value_type)
    )


@compiles(MAP)
def compiles_map(typ, compiler, **kw):
    # TODO: @compiles should live in the dialect
    key_type = compiler.process(typ.key_type, **kw)
    value_type = compiler.process(typ.value_type, **kw)
    return f"MAP({key_type}, {value_type})"


@dt.dtype.register(TrinoDialect, sa.NUMERIC)
def sa_trino_numeric(_, satype, nullable=True):
    return dt.Decimal(satype.precision or 18, satype.scale or 3, nullable=nullable)


@to_sqla_type.register(TrinoDialect, dt.Float64)
def _double(*_):
    return DOUBLE()


@to_sqla_type.register(TrinoDialect, dt.Float32)
def _real(*_):
    return sa.REAL()


@compiles(DOUBLE)
@compiles(sa.REAL, "trino")
def _floating(element, compiler, **kw):
    return type(element).__name__.upper()
