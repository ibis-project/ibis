from __future__ import annotations

from typing import TYPE_CHECKING

import oracledb
import sqlalchemy.types as sat
from sqlalchemy.dialects import oracle

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType

if TYPE_CHECKING:
    from oracle.base_impl import DbType


class OracleType(AlchemyType):
    dialect = "oracle"

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if isinstance(typ, oracle.ROWID):
            return dt.String(nullable=nullable)
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
            return sat.Float(precision=53).with_variant(oracle.FLOAT(14), 'oracle')
        elif isinstance(dtype, dt.Float32):
            return sat.Float(precision=23).with_variant(oracle.FLOAT(7), 'oracle')
        else:
            return super().from_ibis(dtype)


_ORACLE_TYPES = {
    oracledb.DB_TYPE_BINARY_DOUBLE: dt.float64,
    oracledb.DB_TYPE_BINARY_FLOAT: dt.float32,
    oracledb.DB_TYPE_CHAR: dt.string,
    oracledb.DB_TYPE_DATE: dt.date,
    oracledb.DB_TYPE_INTERVAL_DS: dt.Interval,
    oracledb.DB_TYPE_JSON: dt.json,
    oracledb.DB_TYPE_LONG: dt.string,
    oracledb.DB_TYPE_LONG_RAW: dt.bytes,
    oracledb.DB_TYPE_NCHAR: dt.string,
    # this is almost certainly too reductive
    # we'll have already caught the decimals but `NUMBER` can also be a float
    oracledb.DB_TYPE_NUMBER: dt.int,
    oracledb.DB_TYPE_NVARCHAR: dt.string,
    oracledb.DB_TYPE_RAW: dt.bytes,
    oracledb.DB_TYPE_TIMESTAMP: dt.timestamp,
    oracledb.DB_TYPE_TIMESTAMP_LTZ: dt.timestamp,
    oracledb.DB_TYPE_TIMESTAMP_TZ: dt.Timestamp("UTC"),
    oracledb.DB_TYPE_ROWID: dt.string,
    oracledb.DB_TYPE_UROWID: dt.string,
    oracledb.DB_TYPE_VARCHAR: dt.string,
}


def parse(typ: DbType) -> dt.DataType:
    """Parse a Oracle type into an ibis data type."""
    return _ORACLE_TYPES[typ]
