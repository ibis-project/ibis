from __future__ import annotations

from typing import TYPE_CHECKING

import oracledb
import sqlalchemy as sa
from sqlalchemy.dialects import oracle
from sqlalchemy.dialects.oracle.base import OracleDialect
from sqlalchemy.ext.compiler import compiles

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import to_sqla_type

if TYPE_CHECKING:
    from oracle.base_impl import DbType


@dt.dtype.register(OracleDialect, oracle.ROWID)
def sa_oracle_rowid(_, satype, nullable=False):
    return dt.String(nullable=nullable)


@dt.dtype.register(OracleDialect, sa.Numeric)
def sa_oracle_numeric(_, satype, nullable=True):
    if (scale := satype.scale) == 0:
        # kind of a lie, should be int128 because 38 digits
        return dt.Int64(nullable=nullable)
    return dt.Decimal(
        precision=satype.precision or 38,
        scale=scale or 0,
        nullable=nullable,
    )


@dt.dtype.register(OracleDialect, (sa.REAL, sa.FLOAT, sa.Float))
def dtype(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


# @to_sqla_type.register(OracleDialect, dt.Int16)
# def oracle_sa_float16(_, itype):
#     return sa.SmallInteger()


@to_sqla_type.register(OracleDialect, dt.Float64)
def oracle_sa_float64(_, itype):
    # XXX: what should `binary_precision` equal?
    return sa.Float(precision=53).with_variant(
        oracle.FLOAT(binary_precision=14), 'oracle'
    )


@to_sqla_type.register(OracleDialect, dt.Float32)
def oracle_sa_float32(_, itype):
    # XXX: what should `binary_precision` equal?
    return sa.Float(precision=53).with_variant(
        oracle.FLOAT(binary_precision=7), 'oracle'
    )


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


def parse(typ: DbType) -> DataType:
    """Parse a Oracle type into an ibis data type."""

    return _ORACLE_TYPES[typ]
