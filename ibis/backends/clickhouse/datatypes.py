from __future__ import annotations

import functools
from functools import partial
from typing import Literal

import parsy

import ibis
import ibis.expr.datatypes as dt
from ibis.common.parsing import (
    COMMA,
    FIELD,
    LPAREN,
    NUMBER,
    PRECISION,
    RAW_NUMBER,
    RAW_STRING,
    RPAREN,
    SCALE,
    SPACES,
    spaceless_string,
)


def _bool_type() -> Literal["Bool", "UInt8", "Int8"]:
    return getattr(getattr(ibis.options, "clickhouse", None), "bool_type", "Bool")


def parse(text: str) -> dt.DataType:
    datetime64_args = LPAREN.then(
        parsy.seq(
            scale=parsy.decimal_digit.map(int).optional(),
            timezone=COMMA.then(RAW_STRING).optional(),
        )
    ).skip(RPAREN)

    datetime64 = spaceless_string("datetime64").then(
        datetime64_args.optional(default={}).combine_dict(
            partial(dt.Timestamp, nullable=False)
        )
    )

    datetime = spaceless_string("datetime").then(
        parsy.seq(
            timezone=LPAREN.then(RAW_STRING).skip(RPAREN).optional()
        ).combine_dict(partial(dt.Timestamp, nullable=False))
    )

    primitive = (
        datetime64
        | datetime
        | spaceless_string("null", "nothing").result(dt.null)
        | spaceless_string("bigint", "int64").result(dt.Int64(nullable=False))
        | spaceless_string("double", "float64").result(dt.Float64(nullable=False))
        | spaceless_string("float32", "float").result(dt.Float32(nullable=False))
        | spaceless_string("smallint", "int16", "int2").result(dt.Int16(nullable=False))
        | spaceless_string("date32", "date").result(dt.Date(nullable=False))
        | spaceless_string("time").result(dt.Time(nullable=False))
        | spaceless_string("tinyint", "int8", "int1").result(dt.Int8(nullable=False))
        | spaceless_string("boolean", "bool").result(dt.Boolean(nullable=False))
        | spaceless_string("integer", "int32", "int4", "int").result(
            dt.Int32(nullable=False)
        )
        | spaceless_string("uint64").result(dt.UInt64(nullable=False))
        | spaceless_string("uint32").result(dt.UInt32(nullable=False))
        | spaceless_string("uint16").result(dt.UInt16(nullable=False))
        | spaceless_string("uint8").result(dt.UInt8(nullable=False))
        | spaceless_string("uuid").result(dt.UUID(nullable=False))
        | spaceless_string(
            "longtext",
            "mediumtext",
            "tinytext",
            "text",
            "longblob",
            "mediumblob",
            "tinyblob",
            "blob",
            "varchar",
            "char",
            "string",
        ).result(dt.String(nullable=False))
    )

    ty = parsy.forward_declaration()

    nullable = (
        spaceless_string("nullable")
        .then(LPAREN)
        .then(ty.map(lambda ty: ty.copy(nullable=True)))
        .skip(RPAREN)
    )

    fixed_string = (
        spaceless_string("fixedstring")
        .then(LPAREN)
        .then(NUMBER)
        .then(RPAREN)
        .result(dt.String(nullable=False))
    )

    decimal = (
        spaceless_string("decimal", "numeric")
        .then(LPAREN)
        .then(
            parsy.seq(precision=PRECISION.skip(COMMA), scale=SCALE).combine_dict(
                partial(dt.Decimal(nullable=False))
            )
        )
        .skip(RPAREN)
    )

    array = spaceless_string("array").then(
        LPAREN.then(ty.map(partial(dt.Array, nullable=False))).skip(RPAREN)
    )

    map = (
        spaceless_string("map")
        .then(LPAREN)
        .then(parsy.seq(ty, COMMA.then(ty)).combine(partial(dt.Map, nullable=False)))
        .skip(RPAREN)
    )

    at_least_one_space = parsy.regex(r"\s+")

    nested = (
        spaceless_string("nested")
        .then(LPAREN)
        .then(
            parsy.seq(SPACES.then(FIELD.skip(at_least_one_space)), ty)
            .combine(lambda field, ty: (field, dt.Array(ty, nullable=False)))
            .sep_by(COMMA)
            .map(partial(dt.Struct.from_tuples, nullable=False))
        )
        .skip(RPAREN)
    )

    struct = (
        spaceless_string("tuple")
        .then(LPAREN)
        .then(
            parsy.seq(
                SPACES.then(FIELD.skip(at_least_one_space).optional()),
                ty,
            )
            .sep_by(COMMA)
            .map(
                lambda field_names_types: dt.Struct.from_tuples(
                    [
                        (field_name if field_name is not None else f"f{i:d}", typ)
                        for i, (field_name, typ) in enumerate(field_names_types)
                    ],
                    nullable=False,
                )
            )
        )
        .skip(RPAREN)
    )

    enum_value = SPACES.then(RAW_STRING).skip(spaceless_string("=")).then(RAW_NUMBER)

    lowcardinality = (
        spaceless_string("lowcardinality").then(LPAREN).then(ty).skip(RPAREN)
    )

    enum = (
        spaceless_string('enum')
        .then(RAW_NUMBER)
        .then(LPAREN)
        .then(enum_value.sep_by(COMMA))
        .skip(RPAREN)
        .result(dt.String(nullable=False))
    )

    ty.become(
        nullable
        | nested
        | primitive
        | fixed_string
        | decimal
        | array
        | map
        | struct
        | enum
        | lowcardinality
        | spaceless_string("IPv4", "IPv6").result(dt.INET(nullable=False))
        | spaceless_string("Object('json')", "JSON").result(dt.JSON(nullable=False))
    )
    return ty.parse(text)


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
