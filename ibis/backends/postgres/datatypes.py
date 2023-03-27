from __future__ import annotations

import parsy
import sqlalchemy as sa
import toolz
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql.base import PGDialect

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import to_sqla_type
from ibis.common.parsing import (
    COMMA,
    LBRACKET,
    LPAREN,
    PRECISION,
    RBRACKET,
    RPAREN,
    SCALE,
    spaceless,
    spaceless_string,
)

_BRACKETS = "[]"


def _parse_numeric(
    text: str, default_decimal_parameters: tuple[int | None, int | None] = (None, None)
) -> dt.DataType:
    decimal = spaceless_string("decimal", "numeric").then(
        parsy.seq(LPAREN.then(PRECISION.skip(COMMA)), SCALE.skip(RPAREN))
        .optional(default_decimal_parameters)
        .combine(dt.Decimal)
    )

    brackets = spaceless(LBRACKET).then(spaceless(RBRACKET))

    pg_array = parsy.seq(decimal, brackets.at_least(1).map(len)).combine(
        lambda value_type, n: toolz.nth(n, toolz.iterate(dt.Array, value_type))
    )

    ty = pg_array | decimal
    return ty.parse(text)


def _get_type(typestr: str) -> dt.DataType:
    is_array = typestr.endswith(_BRACKETS)
    if (typ := _type_mapping.get(typestr.replace(_BRACKETS, ""))) is not None:
        return dt.Array(typ) if is_array else typ
    try:
        return _parse_numeric(typestr)
    except parsy.ParseError:
        # postgres can have arbitrary types unknown to ibis
        return dt.unknown


_type_mapping = {
    "bigint": dt.int64,
    "boolean": dt.bool,
    "bytea": dt.binary,
    "character varying": dt.string,
    "character": dt.string,
    "character(1)": dt.string,
    "date": dt.date,
    "double precision": dt.float64,
    "geography": dt.geography,
    "geometry": dt.geometry,
    "inet": dt.inet,
    "integer": dt.int32,
    "interval": dt.interval,
    "json": dt.json,
    "jsonb": dt.json,
    "line": dt.linestring,
    "macaddr": dt.macaddr,
    "macaddr8": dt.macaddr,
    "numeric": dt.decimal,
    "point": dt.point,
    "polygon": dt.polygon,
    "real": dt.float32,
    "smallint": dt.int16,
    "text": dt.string,
    # NB: this isn't correct because we're losing the "with time zone"
    # information (ibis doesn't have time type that is time-zone aware), but we
    # try to do _something_ here instead of failing
    "time with time zone": dt.time,
    "time without time zone": dt.time,
    "timestamp with time zone": dt.Timestamp("UTC"),
    "timestamp without time zone": dt.timestamp,
    "uuid": dt.uuid,
}


@to_sqla_type.register(PGDialect, dt.Array)
def _pg_array(dialect, itype):
    # Unwrap the array element type because sqlalchemy doesn't allow arrays of
    # arrays. This doesn't affect the underlying data.
    while itype.is_array():
        itype = itype.value_type
    return sa.ARRAY(to_sqla_type(dialect, itype))


@to_sqla_type.register(PGDialect, dt.Map)
def _pg_map(dialect, itype):
    if not (itype.key_type.is_string() and itype.value_type.is_string()):
        raise TypeError(f"PostgreSQL only supports map<string, string>, got: {itype}")
    return postgresql.HSTORE()


@dt.dtype.register(PGDialect, postgresql.DOUBLE_PRECISION)
def sa_double(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.UUID)
def sa_uuid(_, satype, nullable=True):
    return dt.UUID(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.MACADDR)
def sa_macaddr(_, satype, nullable=True):
    return dt.MACADDR(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.HSTORE)
def sa_hstore(_, satype, nullable=True):
    return dt.Map(dt.string, dt.string, nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.INET)
def sa_inet(_, satype, nullable=True):
    return dt.INET(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.JSONB)
def sa_json(_, satype, nullable=True):
    return dt.JSON(nullable=nullable)


_POSTGRES_FIELD_TO_IBIS_UNIT = {
    "YEAR": "Y",
    "MONTH": "M",
    "DAY": "D",
    "HOUR": "h",
    "MINUTE": "m",
    "SECOND": "s",
    "YEAR TO MONTH": "M",
    "DAY TO HOUR": "h",
    "DAY TO MINUTE": "m",
    "DAY TO SECOND": "s",
    "HOUR TO MINUTE": "m",
    "HOUR TO SECOND": "s",
    "MINUTE TO SECOND": "s",
}


@dt.dtype.register(PGDialect, postgresql.INTERVAL)
def sa_postgres_interval(_, satype, nullable=True):
    field = satype.fields.upper()
    if (unit := _POSTGRES_FIELD_TO_IBIS_UNIT.get(field, None)) is None:
        raise ValueError(f"Unknown PostgreSQL interval field {field!r}")
    elif unit in {"Y", "M"}:
        raise ValueError(
            "Variable length intervals are not yet supported with PostgreSQL"
        )
    return dt.Interval(unit=unit, nullable=nullable)


@dt.dtype.register(PGDialect, sa.ARRAY)
def sa_pg_array(dialect, satype, nullable=True):
    dimensions = satype.dimensions
    if dimensions is not None and dimensions != 1:
        raise NotImplementedError(
            f"Nested array types not yet supported for {dialect.name} dialect"
        )

    value_dtype = dt.dtype(dialect, satype.item_type)
    return dt.Array(value_dtype, nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.TSVECTOR)
def sa_postgres_tsvector(_, satype, nullable=True):
    return dt.Unknown(nullable=nullable)
