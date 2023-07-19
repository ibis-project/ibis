from __future__ import annotations

from functools import partial
from typing import Any

import parsy
import sqlalchemy.types as sat
import trino.client
from sqlalchemy.ext.compiler import compiles
from trino.sqlalchemy.datatype import DOUBLE, JSON, MAP, TIMESTAMP
from trino.sqlalchemy.datatype import ROW as _ROW

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.common.parsing import (
    COMMA,
    FIELD,
    LENGTH,
    LPAREN,
    PRECISION,
    RPAREN,
    SCALE,
    TEMPORAL_SCALE,
    spaceless,
    spaceless_string,
)


class ROW(_ROW):
    _result_is_tuple = hasattr(trino.client, "NamedRowTuple")

    def result_processor(self, dialect, coltype: str) -> None:
        if not coltype.lower().startswith("row"):
            return None

        def process(
            value,
            result_is_tuple: bool = self._result_is_tuple,
            names: tuple[str, ...] = tuple(name for name, _ in self.attr_types),
        ) -> dict[str, Any] | None:
            if value is None or not result_is_tuple:
                return value
            else:
                return dict(zip(names, value))

        return process


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


@compiles(MAP)
def compiles_map(typ, compiler, **kw):
    # TODO: @compiles should live in the dialect
    key_type = compiler.process(typ.key_type, **kw)
    value_type = compiler.process(typ.value_type, **kw)
    return f"MAP({key_type}, {value_type})"


@compiles(DOUBLE)
@compiles(sat.REAL, "trino")
def _floating(element, compiler, **kw):
    return type(element).__name__.upper()


def parse(
    text: str,
    default_decimal_parameters: tuple[int, int] = (18, 3),
    default_temporal_scale: int = 3,  # trino defaults to millisecond scale
) -> dt.DataType:
    """Parse a Trino type into an ibis data type."""

    timestamp = (
        spaceless_string("timestamp")
        .then(
            parsy.seq(
                scale=LPAREN.then(TEMPORAL_SCALE)
                .skip(RPAREN)
                .optional(default_temporal_scale),
                timezone=spaceless_string("with time zone").result("UTC").optional(),
            ).optional(dict(scale=default_temporal_scale, timezone=None))
        )
        .combine_dict(partial(dt.Timestamp))
    )

    primitive = (
        spaceless_string("interval year to month").result(dt.Interval(unit="M"))
        | spaceless_string("interval day to second").result(dt.Interval(unit="ms"))
        | spaceless_string("bigint").result(dt.int64)
        | spaceless_string("boolean").result(dt.boolean)
        | spaceless_string("varbinary").result(dt.binary)
        | spaceless_string("double").result(dt.float64)
        | spaceless_string("real").result(dt.float32)
        | spaceless_string("smallint").result(dt.int16)
        | spaceless_string("date").result(dt.date)
        | spaceless_string("tinyint").result(dt.int8)
        | spaceless_string("integer").result(dt.int32)
        | spaceless_string("uuid").result(dt.uuid)
        | spaceless_string("json").result(dt.json)
        | spaceless_string("ipaddress").result(dt.inet)
    )

    varchar = (
        spaceless_string("varchar", "char")
        .then(LPAREN.then(LENGTH).skip(RPAREN).optional())
        .result(dt.string)
    )

    decimal = spaceless_string("decimal", "numeric").then(
        parsy.seq(LPAREN.then(PRECISION).skip(COMMA), SCALE.skip(RPAREN))
        .optional(default_decimal_parameters)
        .combine(dt.Decimal)
    )

    time = (
        spaceless_string("time").then(
            parsy.seq(
                scale=LPAREN.then(TEMPORAL_SCALE)
                .skip(RPAREN)
                .optional(default_temporal_scale),
                timezone=spaceless_string("with time zone").result("UTC").optional(),
            ).optional(dict(scale=default_temporal_scale, timezone=None))
        )
        # TODO: support time with precision
        .result(dt.time)
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

    ty.become(primitive | timestamp | time | varchar | decimal | array | map | struct)
    return ty.parse(text)


_from_trino_types = {
    DOUBLE: dt.Float64,
    sat.REAL: dt.Float32,
    JSON: dt.JSON,
}


class TrinoType(AlchemyType):
    dialect = "trino"

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if dtype := _from_trino_types.get(type(typ)):
            return dtype(nullable=nullable)
        elif isinstance(typ, sat.NUMERIC):
            return dt.Decimal(typ.precision or 18, typ.scale or 3, nullable=nullable)
        elif isinstance(typ, sat.ARRAY):
            value_dtype = cls.to_ibis(typ.item_type)
            return dt.Array(value_dtype, nullable=nullable)
        elif isinstance(typ, ROW):
            fields = ((k, cls.to_ibis(v)) for k, v in typ.attr_types)
            return dt.Struct.from_tuples(fields, nullable=nullable)
        elif isinstance(typ, MAP):
            return dt.Map(
                cls.to_ibis(typ.key_type),
                cls.to_ibis(typ.value_type),
                nullable=nullable,
            )
        elif isinstance(typ, TIMESTAMP):
            return dt.Timestamp(
                timezone="UTC" if typ.timezone else None,
                scale=typ.precision,
                nullable=nullable,
            )
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        if isinstance(dtype, dt.Float64):
            return DOUBLE()
        elif isinstance(dtype, dt.Float32):
            return sat.REAL()
        elif dtype.is_string():
            return sat.VARCHAR()
        elif dtype.is_struct():
            return ROW((name, cls.from_ibis(typ)) for name, typ in dtype.fields.items())
        elif dtype.is_map():
            return MAP(cls.from_ibis(dtype.key_type), cls.from_ibis(dtype.value_type))
        elif dtype.is_timestamp():
            return TIMESTAMP(precision=dtype.scale, timezone=bool(dtype.timezone))
        else:
            return super().from_ibis(dtype)
