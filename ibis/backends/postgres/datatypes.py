from __future__ import annotations

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as psql
import sqlalchemy.types as sat

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sqlglot.datatypes import PostgresType as SqlglotPostgresType

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

    @classmethod
    def from_string(cls, type_string: str) -> PostgresType:
        return SqlglotPostgresType.from_string(type_string)
