from functools import partial
from typing import Optional, TypedDict

import ibis.expr.datatypes as dt


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
    elif typename == 'FLOAT':
        if col['precision'] <= 24:
            typ = dt.Float32
        else:
            typ = dt.Float64
    return typ(nullable=True)


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
