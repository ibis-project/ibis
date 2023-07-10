from __future__ import annotations

import functools

import sqlalchemy.dialects.postgresql as psql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sqlglot.datatypes import DuckDBType as SqlglotDuckdbType

# _from_duckdb_types = {
#     psql.BYTEA: dt.Binary,
#     psql.UUID: dt.UUID,
#     ducktypes.TinyInteger: dt.Int8,
#     ducktypes.SmallInteger: dt.Int16,
#     ducktypes.Integer: dt.Int32,
#     ducktypes.BigInteger: dt.Int64,
#     ducktypes.HugeInteger: dt.Decimal(38, 0),
#     ducktypes.UInt8: dt.UInt8,
#     ducktypes.UTinyInteger: dt.UInt8,
#     ducktypes.UInt16: dt.UInt16,
#     ducktypes.USmallInteger: dt.UInt16,
#     ducktypes.UInt32: dt.UInt32,
#     ducktypes.UInteger: dt.UInt32,
#     ducktypes.UInt64: dt.UInt64,
#     ducktypes.UBigInteger: dt.UInt64,
# }

# _to_duckdb_types = {
#     dt.UUID: psql.UUID,
#     dt.Int8: ducktypes.TinyInteger,
#     dt.Int16: ducktypes.SmallInteger,
#     dt.Int32: ducktypes.Integer,
#     dt.Int64: ducktypes.BigInteger,
#     dt.UInt8: ducktypes.UTinyInteger,
#     dt.UInt16: ducktypes.USmallInteger,
#     dt.UInt32: ducktypes.UInteger,
#     dt.UInt64: ducktypes.UBigInteger,
# }


class DuckDBType(AlchemyType):
    dialect = "duckdb"

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if dtype := _from_duckdb_types.get(type(typ)):
            return dtype(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        if typ := _to_duckdb_types.get(type(dtype)):
            return typ
        else:
            return super().from_ibis(dtype)

    @classmethod
    def from_string(cls, type_string, nullable=True):
        return SqlglotDuckdbType.from_string(type_string, nullable=nullable)


@functools.singledispatch
def serialize(ty) -> str:
    raise NotImplementedError(f"{ty} not serializable to DuckDB type string")


@serialize.register(dt.DataType)
def _(ty: dt.DataType) -> str:
    ser_ty = serialize_raw(ty)
    if not ty.nullable:
        return f"{ser_ty} NOT NULL"
    return ser_ty


@serialize.register(dt.Map)
def _(ty: dt.Map) -> str:
    return serialize_raw(ty)


@functools.singledispatch
def serialize_raw(ty: dt.DataType) -> str:
    raise NotImplementedError(f"{ty} not serializable to DuckDB type string")


@serialize_raw.register(dt.DataType)
def _(ty: dt.DataType) -> str:
    return type(ty).__name__.capitalize()


@serialize_raw.register(dt.Int8)
def _(_: dt.Int8) -> str:
    return "TINYINT"


@serialize_raw.register(dt.Int16)
def _(_: dt.Int16) -> str:
    return "SMALLINT"


@serialize_raw.register(dt.Int32)
def _(_: dt.Int32) -> str:
    return "INTEGER"


@serialize_raw.register(dt.Int64)
def _(_: dt.Int64) -> str:
    return "BIGINT"


@serialize_raw.register(dt.UInt8)
def _(_: dt.UInt8) -> str:
    return "UTINYINT"


@serialize_raw.register(dt.UInt16)
def _(_: dt.UInt16) -> str:
    return "USMALLINT"


@serialize_raw.register(dt.UInt32)
def _(_: dt.UInt32) -> str:
    return "UINTEGER"


@serialize_raw.register(dt.UInt64)
def _(_: dt.UInt64) -> str:
    return "UBIGINT"


@serialize_raw.register(dt.Float32)
def _(_: dt.Float32) -> str:
    return "FLOAT"


@serialize_raw.register(dt.Float64)
def _(_: dt.Float64) -> str:
    return "DOUBLE"


@serialize_raw.register(dt.Binary)
def _(_: dt.Binary) -> str:
    return "BLOB"


@serialize_raw.register(dt.Boolean)
def _(_: dt.Boolean) -> str:
    return "BOOLEAN"


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
    return f"STRUCT({fields})"


@serialize_raw.register(dt.Timestamp)
def _(ty: dt.Timestamp) -> str:
    if ty.timezone:
        return "TIMESTAMPTZ"
    return "TIMESTAMP"


@serialize_raw.register(dt.Decimal)
def _(ty: dt.Decimal) -> str:
    return f"Decimal({ty.precision}, {ty.scale})"
