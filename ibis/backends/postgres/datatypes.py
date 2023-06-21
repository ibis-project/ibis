from __future__ import annotations

import parsy
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as psql
import sqlalchemy.types as sat
import toolz

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
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


# TODO(kszucs): rename to dtype_from_postgres_typeinfo or parse_postgres_typeinfo
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
    "interval": dt.Interval('s'),
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

_from_postgres_types = {
    psql.DOUBLE_PRECISION: dt.Float64,
    psql.UUID: dt.UUID,
    psql.MACADDR: dt.MACADDR,
    psql.INET: dt.INET,
    psql.JSONB: dt.JSON,
    psql.JSON: dt.JSON,
    psql.TSVECTOR: dt.Unknown,
    psql.BYTEA: dt.Binary,
    psql.UUID: dt.UUID,
}


_postgres_interval_fields = {
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


class PostgresType(AlchemyType):
    dialect = "postgresql"

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> sat.TypeEngine:
        if dtype.is_floating():
            if isinstance(dtype, dt.Float64):
                return psql.DOUBLE_PRECISION
            else:
                return psql.REAL
        elif dtype.is_array():
            # Unwrap the array element type because sqlalchemy doesn't allow arrays of
            # arrays. This doesn't affect the underlying data.
            while dtype.is_array():
                dtype = dtype.value_type
            return sa.ARRAY(cls.from_ibis(dtype))
        elif dtype.is_map():
            if not (dtype.key_type.is_string() and dtype.value_type.is_string()):
                raise TypeError(
                    f"PostgreSQL only supports map<string, string>, got: {dtype}"
                )
            return psql.HSTORE()
        elif dtype.is_uuid():
            return psql.UUID()
        else:
            return super().from_ibis(dtype)

    @classmethod
    def to_ibis(cls, typ: sat.TypeEngine, nullable: bool = True) -> dt.DataType:
        if dtype := _from_postgres_types.get(type(typ)):
            return dtype(nullable=nullable)
        elif isinstance(typ, psql.HSTORE):
            return dt.Map(dt.string, dt.string, nullable=nullable)
        elif isinstance(typ, psql.INTERVAL):
            field = typ.fields.upper()
            if (unit := _postgres_interval_fields.get(field, None)) is None:
                raise ValueError(f"Unknown PostgreSQL interval field {field!r}")
            elif unit in {"Y", "M"}:
                raise ValueError(
                    "Variable length intervals are not yet supported with PostgreSQL"
                )
            return dt.Interval(unit=unit, nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)
