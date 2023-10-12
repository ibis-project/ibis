from __future__ import annotations

import sqlalchemy.types as sat
from sqlalchemy.dialects import oracle

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType
from ibis.backends.base.sqlglot.datatypes import OracleType as SqlglotOracleType


class OracleType(AlchemyType):
    dialect = "oracle"

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if isinstance(typ, oracle.ROWID):
            return dt.String(nullable=nullable)
        elif isinstance(typ, (oracle.RAW, sat.BLOB)):
            return dt.Binary(nullable=nullable)
        elif isinstance(typ, sat.Float):
            return dt.Float64(nullable=nullable)
        elif isinstance(typ, sat.Numeric):
            if typ.scale == 0:
                # kind of a lie, should be int128 because 38 digits
                return dt.Int64(nullable=nullable)
            else:
                return dt.Decimal(
                    precision=typ.precision or 38,
                    scale=typ.scale or 0,
                    nullable=nullable,
                )
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        if isinstance(dtype, dt.Float64):
            return sat.Float(precision=53).with_variant(oracle.FLOAT(14), "oracle")
        elif isinstance(dtype, dt.Float32):
            return sat.Float(precision=23).with_variant(oracle.FLOAT(7), "oracle")
        else:
            return super().from_ibis(dtype)

    @classmethod
    def from_string(cls, type_string, nullable=True):
        return SqlglotOracleType.from_string(type_string, nullable=nullable)
