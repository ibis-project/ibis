from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from snowflake.sqlalchemy import (
    ARRAY,
    OBJECT,
    TIMESTAMP_LTZ,
    TIMESTAMP_NTZ,
    TIMESTAMP_TZ,
    VARIANT,
)
from snowflake.sqlalchemy.snowdialect import SnowflakeDialect

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import to_sqla_type

if TYPE_CHECKING:
    from ibis.expr.datatypes import DataType

_SNOWFLAKE_TYPES = {
    "FIXED": dt.int64,
    "REAL": dt.float64,
    "TEXT": dt.string,
    "DATE": dt.date,
    "TIMESTAMP": dt.timestamp,
    "VARIANT": dt.json,
    "TIMESTAMP_LTZ": dt.timestamp,
    "TIMESTAMP_TZ": dt.Timestamp("UTC"),
    "TIMESTAMP_NTZ": dt.timestamp,
    "OBJECT": dt.Map(dt.string, dt.json),
    "ARRAY": dt.Array(dt.json),
    "BINARY": dt.binary,
    "TIME": dt.time,
    "BOOLEAN": dt.boolean,
}


def parse(text: str) -> DataType:
    """Parse a Snowflake type into an ibis data type."""

    return _SNOWFLAKE_TYPES[text]


@dt.dtype.register(SnowflakeDialect, (TIMESTAMP_LTZ, TIMESTAMP_TZ))
def sa_sf_timestamp_ltz(_, satype, nullable=True):
    return dt.Timestamp(timezone="UTC", nullable=nullable)


@dt.dtype.register(SnowflakeDialect, TIMESTAMP_NTZ)
def sa_sf_timestamp_ntz(_, satype, nullable=True):
    return dt.Timestamp(timezone=None, nullable=nullable)


@dt.dtype.register(SnowflakeDialect, ARRAY)
def sa_sf_array(_, satype, nullable=True):
    return dt.Array(dt.json, nullable=nullable)


@dt.dtype.register(SnowflakeDialect, VARIANT)
def sa_sf_variant(_, satype, nullable=True):
    return dt.JSON(nullable=nullable)


@dt.dtype.register(SnowflakeDialect, OBJECT)
def sa_sf_object(_, satype, nullable=True):
    return dt.Map(dt.string, dt.json, nullable=nullable)


@dt.dtype.register(SnowflakeDialect, sa.Numeric)
def sa_sf_numeric(_, satype, nullable=True):
    if (scale := satype.scale) == 0:
        # kind of a lie, should be int128 because 38 digits
        return dt.Int64(nullable=nullable)
    return dt.Decimal(
        precision=satype.precision or 38,
        scale=scale or 0,
        nullable=nullable,
    )


@dt.dtype.register(SnowflakeDialect, (sa.REAL, sa.FLOAT, sa.Float))
def sa_sf_real_float(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


@to_sqla_type.register(SnowflakeDialect, dt.Array)
def _sf_array(_, itype):
    return ARRAY


@to_sqla_type.register(SnowflakeDialect, (dt.Map, dt.Struct))
def _sf_map_struct(_, itype):
    return OBJECT


@to_sqla_type.register(SnowflakeDialect, dt.JSON)
def _sf_json(_, itype):
    return VARIANT


@to_sqla_type.register(SnowflakeDialect, dt.Timestamp)
def _sf_timestamp(_, itype):
    if itype.timezone is None:
        return TIMESTAMP_NTZ
    return TIMESTAMP_TZ
