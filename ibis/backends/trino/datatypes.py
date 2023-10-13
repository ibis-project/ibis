from __future__ import annotations

from datetime import time, timedelta
from typing import Any

import sqlalchemy.types as sat
import trino.client
from sqlalchemy.ext.compiler import compiles
from trino.sqlalchemy.datatype import DOUBLE, JSON, MAP, TIMESTAMP
from trino.sqlalchemy.datatype import ROW as _ROW

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sqlglot.datatypes import TrinoType as SqlglotTrinoType


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


class INTERVAL(sat.Interval):
    def result_processor(self, dialect, coltype: str) -> None:
        def process(value):
            if value is None:
                return value

            # TODO: support year-month intervals
            days, duration = value.split(" ", 1)
            t = time.fromisoformat(duration)
            return timedelta(
                days=int(days),
                hours=t.hour,
                minutes=t.minute,
                seconds=t.second,
                microseconds=t.microsecond,
            )

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


class TrinoType(AlchemyType):
    dialect = "trino"
    source_types = {
        DOUBLE: dt.Float64,
        sat.REAL: dt.Float32,
        JSON: dt.JSON,
    }

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if dtype := cls.source_types.get(type(typ)):
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

    @classmethod
    def from_string(cls, type_string, nullable=True):
        return SqlglotTrinoType.from_string(type_string, nullable=nullable)
