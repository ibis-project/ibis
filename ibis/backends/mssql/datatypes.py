from __future__ import annotations

from functools import partial
from typing import Optional, TypedDict

from sqlalchemy.dialects import mssql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType


class _FieldDescription(TypedDict):
    """Partial type of result of sp_describe_first_result_set procedure."""

    name: str
    system_type_name: str
    precision: Optional[int]
    scale: Optional[int]


def _type_from_result_set_info(col: _FieldDescription) -> dt.DataType:
    """Construct an ibis type from MSSQL result set description."""
    typename = col["system_type_name"].split("(")[0].upper()
    typ = _from_mssql_typenames.get(typename)
    if typ is None:
        raise NotImplementedError(
            f"MSSQL type {col['system_type_name']} is not supported"
        )

    if typename in ("DECIMAL", "NUMERIC"):
        typ = partial(typ, precision=col["precision"], scale=col["scale"])
    elif typename in ("GEOMETRY", "GEOGRAPHY"):
        typ = partial(typ, geotype=typename.lower())
    elif typename == "DATETIME2":
        typ = partial(typ, scale=col["scale"])
    elif typename == "DATETIMEOFFSET":
        typ = partial(typ, scale=col["scale"], timezone="UTC")
    elif typename == "FLOAT":
        if col["precision"] <= 24:
            typ = dt.Float32
        else:
            typ = dt.Float64
    return typ(nullable=col["is_nullable"])


# The following MSSQL 2022 types are not supported: 'XML', 'SQL_VARIANT', 'SYSNAME', 'HIERARCHYID',
_from_mssql_typenames = {
    # Exact numerics
    "BIGINT": dt.Int64,
    "BIT": dt.Boolean,
    "DECIMAL": dt.Decimal,
    "INT": dt.Int32,
    "MONEY": dt.Int64,
    "NUMERIC": dt.Decimal,
    "SMALLINT": dt.Int16,
    "SMALLMONEY": dt.Int32,
    "TINYINT": dt.Int8,
    # Approximate numerics
    "FLOAT": dt.Float64,
    "REAL": dt.Float32,
    # Date and time
    "DATE": dt.Date,
    "DATETIME2": dt.Timestamp,
    "DATETIME": dt.Timestamp,
    "DATETIMEOFFSET": dt.Timestamp,
    "SMALLDATETIME": dt.Timestamp,
    "TIME": dt.Time,
    # Character string
    "CHAR": dt.String,
    "TEXT": dt.String,
    "VARCHAR": dt.String,
    # Unicode character strings
    "NCHAR": dt.String,
    "NTEXT": dt.String,
    "NVARCHAR": dt.String,
    # Binary string
    "BINARY": dt.Binary,
    "IMAGE": dt.Binary,
    "VARBINARY": dt.Binary,
    # Other data types
    "UNIQUEIDENTIFIER": dt.UUID,
    "GEOMETRY": dt.GeoSpatial,
    "GEOGRAPHY": dt.GeoSpatial,
    # This timestamp datatype is also known as "rowversion", and the original name is really unfortunate.
    # See:
    # https://learn.microsoft.com/en-us/sql/t-sql/data-types/rowversion-transact-sql?view=sql-server-ver16
    "TIMESTAMP": dt.Binary,
}


_to_mssql_types = {
    dt.Boolean: mssql.BIT,
    dt.Int8: mssql.TINYINT,
    dt.Int16: mssql.SMALLINT,
    dt.Int32: mssql.INTEGER,
    dt.Int64: mssql.BIGINT,
    dt.Float16: mssql.FLOAT,
    dt.Float32: mssql.FLOAT,
    dt.Float64: mssql.REAL,
    dt.String: mssql.NVARCHAR,
}

_from_mssql_types = {
    mssql.TINYINT: dt.Int8,
    mssql.BIT: dt.Boolean,
    mssql.MONEY: dt.Int64,
    mssql.SMALLMONEY: dt.Int32,
    mssql.UNIQUEIDENTIFIER: dt.UUID,
    mssql.BINARY: dt.Binary,
    mssql.TIMESTAMP: dt.Binary,
    mssql.NVARCHAR: dt.String,
    mssql.NTEXT: dt.String,
    mssql.VARBINARY: dt.Binary,
    mssql.IMAGE: dt.Binary,
    mssql.TIME: dt.Time,
    mssql.NCHAR: dt.String,
}


class MSSQLType(AlchemyType):
    dialect = "mssql"

    @classmethod
    def to_ibis(cls, typ, nullable=True):
        if dtype := _from_mssql_types.get(type(typ)):
            return dtype(nullable=nullable)
        elif isinstance(typ, mssql.DATETIMEOFFSET):
            if (prec := typ.precision) is None:
                prec = 7
            return dt.Timestamp(scale=prec, timezone="UTC", nullable=nullable)
        elif isinstance(typ, mssql.DATETIME2):
            if (prec := typ.precision) is None:
                prec = 7
            return dt.Timestamp(scale=prec, nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype):
        if typ := _to_mssql_types.get(type(dtype)):
            return typ
        elif dtype.is_timestamp():
            if (precision := dtype.scale) is None:
                precision = 7
            if dtype.timezone is not None:
                return mssql.DATETIMEOFFSET(precision=precision)
            else:
                return mssql.DATETIME2(precision=precision)
        else:
            return super().from_ibis(dtype)
