from __future__ import annotations

import parsy as p
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
    inet,
    int8,
    int16,
    int32,
    int64,
    json,
    string,
    time,
    uuid,
)


def parse(text: str, default_decimal_parameters=(18, 3)) -> DataType:
    """Parse a Trino type into an ibis data type."""

    @p.generate
    def timestamp():
        yield spaceless_string("timestamp")
        yield LPAREN.then(SINGLE_DIGIT.map(int)).skip(RPAREN).optional()
        return Timestamp(timezone="UTC")

    primitive = (
        spaceless_string("interval").result(Interval())
        | spaceless_string("bigint").result(int64)
        | spaceless_string("boolean").result(boolean)
        | spaceless_string("varbinary").result(binary)
        | spaceless_string("double").result(float64)
        | spaceless_string("real").result(float32)
        | spaceless_string("smallint").result(int16)
        | timestamp
        | spaceless_string("date").result(date)
        | spaceless_string("time").result(time)
        | spaceless_string("tinyint").result(int8)
        | spaceless_string("integer").result(int32)
        | spaceless_string("uuid").result(uuid)
        | spaceless_string("varchar", "char").result(string)
        | spaceless_string("json").result(json)
        | spaceless_string("ipaddress").result(inet)
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
        yield LPAREN
        value_type = yield ty
        yield RPAREN
        return value_type

    @p.generate
    def array():
        yield spaceless_string("array")
        value_type = yield angle_type
        return Array(value_type)

    @p.generate
    def map():
        yield spaceless_string("map")
        yield LPAREN
        key_type = yield primitive
        yield COMMA
        value_type = yield ty
        yield RPAREN
        return Map(key_type, value_type)

    field = spaceless(FIELD)

    @p.generate
    def struct():
        yield spaceless_string("row")
        yield LPAREN
        field_names_types = yield (
            p.seq(field, ty).combine(lambda field, ty: (field, ty)).sep_by(COMMA)
        )
        yield RPAREN
        return Struct.from_tuples(field_names_types)

    ty = primitive | decimal | array | map | struct
    return ty.parse(text)


@dt.dtype.register(TrinoDialect, DOUBLE)
def sa_trino_double(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


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


@compiles(TIMESTAMP, "trino")
def compiles_timestamp(typ, compiler, **kw):
    result = "TIMESTAMP"

    if (prec := typ.precision) is not None:
        result += f"({prec:d})"

    if typ.timezone:
        result += " WITH TIME ZONE"

    return result


@compiles(ROW, "trino")
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


@compiles(MAP, "trino")
def compiles_map(typ, compiler, **kw):
    # TODO: @compiles should live in the dialect
    key_type = compiler.process(typ.key_type, **kw)
    value_type = compiler.process(typ.value_type, **kw)
    return f"MAP({key_type}, {value_type})"


@dt.dtype.register(TrinoDialect, sa.NUMERIC)
def sa_trino_numeric(_, satype, nullable=True):
    return dt.Decimal(satype.precision or 18, satype.scale or 3, nullable=nullable)
