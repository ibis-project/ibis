from functools import partial
from typing import Optional, TypedDict

from sqlalchemy.dialects import mssql
from sqlalchemy.dialects.mssql.base import MSDialect

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import to_sqla_type


class _FieldDescription(TypedDict):
    """Partial type of result of sp_describe_first_result_set procedure."""

    name: str
    system_type_name: str
    precision: Optional[int]
    scale: Optional[int]


def _type_from_result_set_info(col: _FieldDescription) -> dt.DataType:
    """Construct an ibis type from MSSQL result set description."""
    typename = col['system_type_name'].split('(')[0].upper()
    typ = _type_mapping.get(typename)
    if typ is None:
        raise NotImplementedError(
            f"MSSQL type {col['system_type_name']} is not supported"
        )

    if typename in ("DECIMAL", "NUMERIC"):
        typ = partial(typ, precision=col["precision"], scale=col['scale'])
    elif typename in ("GEOMETRY", "GEOGRAPHY"):
        typ = partial(typ, geotype=typename.lower())
    elif typename == 'DATETIME2':
        typ = partial(typ, scale=col["scale"])
    elif typename == 'DATETIMEOFFSET':
        typ = partial(typ, scale=col["scale"], timezone="UTC")
    elif typename == 'FLOAT':
        if col['precision'] <= 24:
            typ = dt.Float32
        else:
            typ = dt.Float64
    return typ(nullable=col["is_nullable"])


# The following MSSQL 2022 types are not supported: 'XML', 'SQL_VARIANT', 'SYSNAME', 'HIERARCHYID',
_type_mapping = {
    # Exact numerics
    'BIGINT': dt.Int64,
    'BIT': dt.Boolean,
    'DECIMAL': dt.Decimal,
    'INT': dt.Int32,
    'MONEY': dt.Int64,
    'NUMERIC': dt.Decimal,
    'SMALLINT': dt.Int16,
    'SMALLMONEY': dt.Int32,
    'TINYINT': dt.Int8,
    # Approximate numerics
    'FLOAT': dt.Float64,
    'REAL': dt.Float32,
    # Date and time
    'DATE': dt.Date,
    'DATETIME2': dt.Timestamp,
    'DATETIME': dt.Timestamp,
    'DATETIMEOFFSET': dt.Timestamp,
    'SMALLDATETIME': dt.Timestamp,
    'TIME': dt.Time,
    # Character string
    'CHAR': dt.String,
    'TEXT': dt.String,
    'VARCHAR': dt.String,
    # Unicode character strings
    'NCHAR': dt.String,
    'NTEXT': dt.String,
    'NVARCHAR': dt.String,
    # Binary string
    'BINARY': dt.Binary,
    'IMAGE': dt.Binary,
    'VARBINARY': dt.Binary,
    # Other data types
    'UNIQUEIDENTIFIER': dt.UUID,
    'GEOMETRY': dt.GeoSpatial,
    'GEOGRAPHY': dt.GeoSpatial,
    # This timestamp datatype is also known as "rowversion", and the original name is really unfortunate.
    # See:
    # https://learn.microsoft.com/en-us/sql/t-sql/data-types/rowversion-transact-sql?view=sql-server-ver16
    'TIMESTAMP': dt.Binary,
}


_MSSQL_TYPE_MAP = {
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


@to_sqla_type.register(mssql.dialect, tuple(_MSSQL_TYPE_MAP.keys()))
def _simple_types(_, itype):
    return _MSSQL_TYPE_MAP[type(itype)]


@to_sqla_type.register(mssql.dialect, dt.Timestamp)
def _datetime(_, itype):
    if (precision := itype.scale) is None:
        precision = 7
    if itype.timezone is not None:
        return mssql.DATETIMEOFFSET(precision=precision)
    else:
        return mssql.DATETIME2(precision=precision)


@dt.dtype.register(MSDialect, mssql.TINYINT)
def sa_mysql_tinyint(_, satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(MSDialect, mssql.BIT)
def sa_mssql_bit(_, satype, nullable=True):
    return dt.Boolean(nullable=nullable)


@dt.dtype.register(MSDialect, mssql.MONEY)
def sa_bigint(_, satype, nullable=True):
    return dt.Int64(nullable=nullable)


@dt.dtype.register(MSDialect, mssql.SMALLMONEY)
def sa_mssql_smallmoney(_, satype, nullable=True):
    return dt.Int32(nullable=nullable)


@dt.dtype.register(MSDialect, mssql.UNIQUEIDENTIFIER)
def sa_mssql_uuid(_, satype, nullable=True):
    return dt.UUID(nullable=nullable)


@dt.dtype.register(MSDialect, (mssql.BINARY, mssql.TIMESTAMP))
def sa_mssql_binary(_, satype, nullable=True):
    return dt.Binary(nullable=nullable)


@dt.dtype.register(MSDialect, mssql.DATETIMEOFFSET)
def _datetimeoffset(_, sa_type, nullable=True):
    if (prec := sa_type.precision) is None:
        prec = 7
    return dt.Timestamp(scale=prec, timezone="UTC", nullable=nullable)


@dt.dtype.register(MSDialect, mssql.DATETIME2)
def _datetime2(_, sa_type, nullable=True):
    if (prec := sa_type.precision) is None:
        prec = 7
    return dt.Timestamp(scale=prec, nullable=nullable)
