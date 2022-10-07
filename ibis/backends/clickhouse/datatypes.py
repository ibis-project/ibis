from __future__ import annotations

import functools

import parsy

import ibis.expr.datatypes as dt
from ibis.common.parsing import (
    COMMA,
    FIELD,
    LPAREN,
    NUMBER,
    PRECISION,
    RAW_STRING,
    RPAREN,
    SCALE,
    SPACES,
    spaceless_string,
)


def parse(text: str) -> dt.DataType:
    @parsy.generate
    def datetime():
        yield spaceless_string("datetime64", "datetime")
        timezone = yield parened_string.optional()
        return dt.Timestamp(timezone=timezone, nullable=False)

    primitive = (
        datetime
        | spaceless_string("null", "nothing").result(dt.null)
        | spaceless_string("bigint", "int64").result(dt.Int64(nullable=False))
        | spaceless_string("double", "float64").result(dt.Float64(nullable=False))
        | spaceless_string("float32", "float").result(dt.Float32(nullable=False))
        | spaceless_string("smallint", "int16", "int2").result(dt.Int16(nullable=False))
        | spaceless_string("date32", "date").result(dt.Date(nullable=False))
        | spaceless_string("time").result(dt.Time(nullable=False))
        | spaceless_string("tinyint", "int8", "int1", "boolean", "bool").result(
            dt.Int8(nullable=False)
        )
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

    @parsy.generate
    def parened_string():
        yield LPAREN
        s = yield RAW_STRING
        yield RPAREN
        return s

    @parsy.generate
    def nullable():
        yield spaceless_string("nullable")
        yield LPAREN
        parsed_ty = yield ty
        yield RPAREN
        return parsed_ty(nullable=True)

    @parsy.generate
    def fixed_string():
        yield spaceless_string("fixedstring")
        yield LPAREN
        yield NUMBER
        yield RPAREN
        return dt.String(nullable=False)

    @parsy.generate
    def decimal():
        yield spaceless_string("decimal", "numeric")
        precision, scale = yield LPAREN.then(
            parsy.seq(PRECISION.skip(COMMA), SCALE)
        ).skip(RPAREN)
        return dt.Decimal(precision, scale, nullable=False)

    @parsy.generate
    def paren_type():
        yield LPAREN
        value_type = yield ty
        yield RPAREN
        return value_type

    @parsy.generate
    def array():
        yield spaceless_string("array")
        value_type = yield paren_type
        return dt.Array(value_type, nullable=False)

    @parsy.generate
    def map():
        yield spaceless_string("map")
        yield LPAREN
        key_type = yield ty
        yield COMMA
        value_type = yield ty
        yield RPAREN
        return dt.Map(key_type, value_type, nullable=False)

    at_least_one_space = parsy.regex(r"\s+")

    @parsy.generate
    def nested():
        yield spaceless_string("nested")
        yield LPAREN

        field_names_types = yield (
            parsy.seq(SPACES.then(FIELD.skip(at_least_one_space)), ty)
            .combine(lambda field, ty: (field, dt.Array(ty, nullable=False)))
            .sep_by(COMMA)
        )
        yield RPAREN
        return dt.Struct.from_tuples(field_names_types, nullable=False)

    @parsy.generate
    def struct():
        yield spaceless_string("tuple")
        yield LPAREN
        field_names_types = yield (
            parsy.seq(
                SPACES.then(FIELD.skip(at_least_one_space).optional()),
                ty,
            )
            .combine(lambda field, ty: (field, ty))
            .sep_by(COMMA)
        )
        yield RPAREN
        return dt.Struct.from_tuples(
            [
                (field_name if field_name is not None else f"f{i:d}", typ)
                for i, (field_name, typ) in enumerate(field_names_types)
            ],
            nullable=False,
        )

    @parsy.generate
    def enum_value():
        yield SPACES
        key = yield RAW_STRING
        yield spaceless_string('=')
        value = yield parsy.digit.at_least(1).concat()
        return (key, int(value))

    @parsy.generate
    def lowcardinality():
        yield spaceless_string('LowCardinality')
        yield LPAREN
        r = yield ty
        yield RPAREN
        return r

    @parsy.generate
    def enum():
        yield spaceless_string('enum')
        enumsz = yield parsy.digit.at_least(1).concat()
        enumsz = int(enumsz)
        yield LPAREN
        yield enum_value.sep_by(COMMA).map(dict)  # ignore values
        yield RPAREN
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
        | spaceless_string("IPv4", "IPv6").result(dt.inet(nullable=False))
        | spaceless_string("Object('json')").result(dt.json(nullable=False))
        | spaceless_string("JSON").result(dt.json(nullable=False))
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
        f"{name} {serialize(field_ty)}" for name, field_ty in ty.pairs.items()
    )
    return f"Tuple({fields})"


@serialize_raw.register(dt.Timestamp)
def _(ty: dt.Timestamp) -> str:
    return "DateTime64(6)" if ty.timezone is None else f"DateTime64(6, {ty.timezone!r})"
