from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import sqlglot as sg
from sqlglot.expressions import ColumnDef, DataType

import ibis
import ibis.expr.datatypes as dt
from ibis.common.collections import FrozenDict
from ibis.formats.parser import TypeParser

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlglot.expressions import Expression

    try:
        from sqlglot.expressions import DataTypeParam
    except ImportError:
        from sqlglot.expressions import DataTypeSize as DataTypeParam


def _bool_type() -> Literal["Bool", "UInt8", "Int8"]:
    return getattr(getattr(ibis.options, "clickhouse", None), "bool_type", "Bool")


class ClickHouseTypeParser(TypeParser):
    __slots__ = ()

    dialect = "clickhouse"
    default_decimal_precision = None
    default_decimal_scale = None
    default_nullable = False

    short_circuit: Mapping[str, dt.DataType] = FrozenDict(
        {
            "IPv4": dt.INET(nullable=default_nullable),
            "IPv6": dt.INET(nullable=default_nullable),
            "Object('json')": dt.JSON(nullable=default_nullable),
            "Array(Null)": dt.Array(dt.null, nullable=default_nullable),
            "Array(Nothing)": dt.Array(dt.null, nullable=default_nullable),
        }
    )

    @classmethod
    def _get_DATETIME(
        cls, first: DataTypeParam | None = None, second: DataTypeParam | None = None
    ) -> dt.Timestamp:
        if first is not None and second is not None:
            scale = first
            timezone = second
        elif first is not None and second is None:
            timezone, scale = (
                (first, second) if first.this.is_string else (second, first)
            )
        else:
            scale = first
            timezone = second
        return cls._get_TIMESTAMP(scale=scale, timezone=timezone)

    @classmethod
    def _get_DATETIME64(
        cls, scale: DataTypeParam | None = None, timezone: DataTypeParam | None = None
    ) -> dt.Timestamp:
        return cls._get_TIMESTAMP(scale=scale, timezone=timezone)

    @classmethod
    def _get_NULLABLE(cls, inner_type: DataType) -> dt.DataType:
        return cls._get_type(inner_type).copy(nullable=True)

    @classmethod
    def _get_LOWCARDINALITY(cls, inner_type: DataType) -> dt.DataType:
        return cls._get_type(inner_type)

    @classmethod
    def _get_NESTED(cls, *fields: DataType) -> dt.Struct:
        return dt.Struct(
            {
                field.name: dt.Array(
                    cls._get_type(field.args["kind"]), nullable=cls.default_nullable
                )
                for field in fields
            },
            nullable=cls.default_nullable,
        )

    @classmethod
    def _get_STRUCT(cls, *fields: Expression) -> dt.Struct:
        types = {}

        for i, field in enumerate(fields):
            if isinstance(field, ColumnDef):
                inner_type = field.args["kind"]
                name = field.name
            else:
                inner_type = sg.parse_one(str(field), into=DataType, read="clickhouse")
                name = f"f{i:d}"

            types[name] = cls._get_type(inner_type)
        return dt.Struct(types, nullable=cls.default_nullable)


parse = ClickHouseTypeParser.parse


@functools.singledispatch
def serialize(ty) -> str:
    raise NotImplementedError(f"{ty} not serializable to clickhouse type string")


@serialize.register(dt.DataType)
def _(ty: dt.DataType) -> str:
    ser_ty = serialize_raw(ty)
    if ty.nullable:
        return f"Nullable({ser_ty})"
    return ser_ty


@serialize.register(dt.Map)
def _(ty: dt.Map) -> str:
    return serialize_raw(ty)


@functools.singledispatch
def serialize_raw(ty: dt.DataType) -> str:
    raise NotImplementedError(f"{ty} not serializable to clickhouse type string")


@serialize_raw.register(dt.DataType)
def _(ty: dt.DataType) -> str:
    return type(ty).__name__.capitalize()


@serialize_raw.register(dt.Binary)
def _(_: dt.Binary) -> str:
    return "BLOB"


@serialize_raw.register(dt.Boolean)
def _(_: dt.Boolean) -> str:
    return _bool_type()


@serialize_raw.register(dt.Array)
def _(ty: dt.Array) -> str:
    return f"Array({serialize(ty.value_type)})"


@serialize_raw.register(dt.Map)
def _(ty: dt.Map) -> str:
    # nullable key type is not allowed inside maps
    key_type = serialize_raw(ty.key_type)
    value_type = serialize(ty.value_type)
    return f"Map({key_type}, {value_type})"


@serialize_raw.register(dt.Struct)
def _(ty: dt.Struct) -> str:
    fields = ", ".join(
        f"{name} {serialize(field_ty)}" for name, field_ty in ty.fields.items()
    )
    return f"Tuple({fields})"


@serialize_raw.register(dt.Timestamp)
def _(ty: dt.Timestamp) -> str:
    if (scale := ty.scale) is None:
        scale = 3

    if (timezone := ty.timezone) is not None:
        return f"DateTime64({scale:d}, {timezone!r})"
    return f"DateTime64({scale:d})"


@serialize_raw.register(dt.Decimal)
def _(ty: dt.Decimal) -> str:
    return f"Decimal({ty.precision}, {ty.scale})"
