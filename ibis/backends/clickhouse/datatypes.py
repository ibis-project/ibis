from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import parsy as p

if TYPE_CHECKING:
    from ibis.expr.datatypes import DataType

import ibis.expr.datatypes as dt


def parse(text: str) -> DataType:
    @p.generate
    def datetime():
        yield dt.spaceless_string("datetime64", "datetime")
        timezone = yield parened_string.optional()
        return dt.Timestamp(timezone=timezone, nullable=False)

    primitive = (
        datetime
        | dt.spaceless_string("null", "nothing").result(dt.null)
        | dt.spaceless_string("bigint", "int64").result(
            dt.Int64(nullable=False)
        )
        | dt.spaceless_string("double", "float64").result(
            dt.Float64(nullable=False)
        )
        | dt.spaceless_string("float32", "float").result(
            dt.Float32(nullable=False)
        )
        | dt.spaceless_string("smallint", "int16", "int2").result(
            dt.Int16(nullable=False)
        )
        | dt.spaceless_string("date32", "date").result(dt.Date(nullable=False))
        | dt.spaceless_string("time").result(dt.Time(nullable=False))
        | dt.spaceless_string(
            "tinyint", "int8", "int1", "boolean", "bool"
        ).result(dt.Int8(nullable=False))
        | dt.spaceless_string("integer", "int32", "int4", "int").result(
            dt.Int32(nullable=False)
        )
        | dt.spaceless_string("uint64").result(dt.UInt64(nullable=False))
        | dt.spaceless_string("uint32").result(dt.UInt32(nullable=False))
        | dt.spaceless_string("uint16").result(dt.UInt16(nullable=False))
        | dt.spaceless_string("uint8").result(dt.UInt8(nullable=False))
        | dt.spaceless_string("uuid").result(dt.UUID(nullable=False))
        | dt.spaceless_string(
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

    @p.generate
    def parened_string():
        yield dt.LPAREN
        s = yield dt.RAW_STRING
        yield dt.RPAREN
        return s

    @p.generate
    def nullable():
        yield dt.spaceless_string("nullable")
        yield dt.LPAREN
        parsed_ty = yield ty
        yield dt.RPAREN
        return parsed_ty(nullable=True)

    @p.generate
    def fixed_string():
        yield dt.spaceless_string("fixedstring")
        yield dt.LPAREN
        yield dt.NUMBER
        yield dt.RPAREN
        return dt.String(nullable=False)

    @p.generate
    def decimal():
        yield dt.spaceless_string("decimal", "numeric")
        precision, scale = yield dt.LPAREN.then(
            p.seq(dt.PRECISION.skip(dt.COMMA), dt.SCALE)
        ).skip(dt.RPAREN)
        return dt.Decimal(precision, scale, nullable=False)

    @p.generate
    def paren_type():
        yield dt.LPAREN
        value_type = yield ty
        yield dt.RPAREN
        return value_type

    @p.generate
    def array():
        yield dt.spaceless_string("array")
        value_type = yield paren_type
        return dt.Array(value_type, nullable=False)

    @p.generate
    def map():
        yield dt.spaceless_string("map")
        yield dt.LPAREN
        key_type = yield ty
        yield dt.COMMA
        value_type = yield ty
        yield dt.RPAREN
        return dt.Map(key_type, value_type, nullable=False)

    at_least_one_space = p.regex(r"\s+")

    @p.generate
    def nested():
        yield dt.spaceless_string("nested")
        yield dt.LPAREN

        field_names_types = yield (
            p.seq(dt.SPACES.then(dt.FIELD.skip(at_least_one_space)), ty)
            .combine(lambda field, ty: (field, dt.Array(ty, nullable=False)))
            .sep_by(dt.COMMA)
        )
        yield dt.RPAREN
        return dt.Struct.from_tuples(field_names_types, nullable=False)

    @p.generate
    def struct():
        yield dt.spaceless_string("tuple")
        yield dt.LPAREN
        field_names_types = yield (
            p.seq(
                dt.SPACES.then(dt.FIELD.skip(at_least_one_space).optional()),
                ty,
            )
            .combine(lambda field, ty: (field, ty))
            .sep_by(dt.COMMA)
        )
        yield dt.RPAREN
        return dt.Struct.from_tuples(
            [
                (field_name if field_name is not None else f"f{i:d}", typ)
                for i, (field_name, typ) in enumerate(field_names_types)
            ],
            nullable=False,
        )

    @p.generate
    def enum_value():
        yield dt.SPACES
        key = yield dt.RAW_STRING
        yield dt.spaceless_string('=')
        value = yield p.digit.at_least(1).concat()
        return (key, int(value))

    @p.generate
    def lowcardinality():
        yield dt.spaceless_string('LowCardinality')
        yield dt.LPAREN
        r = yield ty
        yield dt.RPAREN
        return r

    @p.generate
    def enum():
        yield dt.spaceless_string('enum')
        enumsz = yield p.digit.at_least(1).concat()
        enumsz = int(enumsz)
        yield dt.LPAREN
        yield enum_value.sep_by(dt.COMMA).map(dict)  # ignore values
        yield dt.RPAREN
        return dt.String(nullable=False)

    ty = (
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
        | dt.spaceless_string("IPv4", "IPv6").result(dt.inet(nullable=False))
        | dt.spaceless_string("Object('json')").result(dt.json(nullable=False))
        | dt.spaceless_string("JSON").result(dt.json(nullable=False))
    )
    return ty.parse(text)


@functools.singledispatch
def serialize(ty) -> str:
    raise NotImplementedError(
        f"{ty} not serializable to clickhouse type string"
    )


@serialize.register(dt.DataType)
def _(ty: dt.DataType) -> str:
    ser_ty = serialize_raw(ty)
    if ty.nullable:
        return f"Nullable({ser_ty})"
    return ser_ty


@functools.singledispatch
def serialize_raw(ty: dt.DataType) -> str:
    raise NotImplementedError(
        f"{ty} not serializable to clickhouse type string"
    )


@serialize_raw.register(dt.DataType)
def _(ty: dt.DataType) -> str:
    return type(ty).__name__.capitalize()


@serialize_raw.register(dt.Array)
def _(ty: dt.Array) -> str:
    return f"Array({serialize(ty.value_type)})"


@serialize_raw.register(dt.Map)
def _(ty: dt.Map) -> str:
    key_type = serialize(ty.key_type)
    value_type = serialize(ty.value_type)
    return f"Map({key_type}, {value_type})"


@serialize_raw.register(dt.Struct)
def _(ty: dt.Struct) -> str:
    fields = ", ".join(
        f"{name} {serialize(field_ty)}" for name, field_ty in ty.pairs.items()
    )
    return f"Tuple({fields})"


@serialize_raw.register(dt.Timestamp)
def _(ty: dt.Timestamp) -> str:
    return (
        "DateTime64(6)"
        if ty.timezone is None
        else f"DateTime64(6, {ty.timezone!r})"
    )
