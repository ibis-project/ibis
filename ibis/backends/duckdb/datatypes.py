from __future__ import annotations

import duckdb_engine.datatypes as ducktypes
import sqlalchemy.dialects.postgresql as psql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sqlglot.datatypes import DuckDBType as SqlglotDuckdbType

_from_duckdb_types = {
    psql.BYTEA: dt.Binary,
    psql.UUID: dt.UUID,
    ducktypes.TinyInteger: dt.Int8,
    ducktypes.SmallInteger: dt.Int16,
    ducktypes.Integer: dt.Int32,
    ducktypes.BigInteger: dt.Int64,
    ducktypes.HugeInteger: dt.Decimal(38, 0),
    ducktypes.UInt8: dt.UInt8,
    ducktypes.UTinyInteger: dt.UInt8,
    ducktypes.UInt16: dt.UInt16,
    ducktypes.USmallInteger: dt.UInt16,
    ducktypes.UInt32: dt.UInt32,
    ducktypes.UInteger: dt.UInt32,
    ducktypes.UInt64: dt.UInt64,
    ducktypes.UBigInteger: dt.UInt64,
}

_to_duckdb_types = {
    dt.UUID: psql.UUID,
    dt.Int8: ducktypes.TinyInteger,
    dt.Int16: ducktypes.SmallInteger,
    dt.Int32: ducktypes.Integer,
    dt.Int64: ducktypes.BigInteger,
    dt.UInt8: ducktypes.UTinyInteger,
    dt.UInt16: ducktypes.USmallInteger,
    dt.UInt32: ducktypes.UInteger,
    dt.UInt64: ducktypes.UBigInteger,
}


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
