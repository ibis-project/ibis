from __future__ import annotations

from typing import Iterable

import sqlalchemy as sa
from multipledispatch import Dispatcher
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite
from sqlalchemy.dialects.mssql.base import MSDialect
from sqlalchemy.dialects.mysql.base import MySQLDialect
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from sqlalchemy.engine.default import DefaultDialect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import UserDefinedType

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported

if geospatial_supported:
    import geoalchemy2 as ga


class ArrayType(UserDefinedType):
    def __init__(self, value_type: sa.types.TypeEngine):
        self.value_type = sa.types.to_instance(value_type)


@compiles(ArrayType, "default")
def compiles_array(element, compiler, **kw):
    return f"ARRAY({compiler.process(element.value_type, **kw)})"


class StructType(UserDefinedType):
    def __init__(
        self,
        pairs: Iterable[tuple[str, sa.types.TypeEngine]],
    ):
        self.pairs = [(name, sa.types.to_instance(type)) for name, type in pairs]


@compiles(StructType, "default")
def compiles_struct(element, compiler, **kw):
    content = ", ".join(
        f"{field} {compiler.process(typ, **kw)}" for field, typ in element.pairs
    )
    return f"STRUCT({content})"


class MapType(UserDefinedType):
    def __init__(self, key_type: sa.types.TypeEngine, value_type: sa.types.TypeEngine):
        self.key_type = sa.types.to_instance(key_type)
        self.value_type = sa.types.to_instance(value_type)


@compiles(MapType, "default")
def compiles_map(element, compiler, **kw):
    key_type = compiler.process(element.key_type, **kw)
    value_type = compiler.process(element.value_type, **kw)
    return f"MAP({key_type}, {value_type})"


class UInt64(sa.types.Integer):
    pass


class UInt32(sa.types.Integer):
    pass


class UInt16(sa.types.Integer):
    pass


class UInt8(sa.types.Integer):
    pass


@compiles(UInt64, "postgresql")
@compiles(UInt32, "postgresql")
@compiles(UInt16, "postgresql")
@compiles(UInt8, "postgresql")
@compiles(UInt64, "mysql")
@compiles(UInt32, "mysql")
@compiles(UInt16, "mysql")
@compiles(UInt8, "mysql")
@compiles(UInt64, "sqlite")
@compiles(UInt32, "sqlite")
@compiles(UInt16, "sqlite")
@compiles(UInt8, "sqlite")
def compile_uint(element, compiler, **kw):
    dialect_name = compiler.dialect.name
    raise TypeError(
        f"unsigned integers are not supported in the {dialect_name} backend"
    )


def table_from_schema(name, meta, schema, database: str | None = None):
    # Convert Ibis schema to SQLA table
    columns = []

    dialect = getattr(meta.bind, "dialect", _DEFAULT_DIALECT)
    for colname, dtype in zip(schema.names, schema.types):
        satype = to_sqla_type(dialect, dtype)
        column = sa.Column(colname, satype, nullable=dtype.nullable)
        columns.append(column)

    return sa.Table(name, meta, *columns, schema=database)


# TODO(cleanup)
ibis_type_to_sqla = {
    dt.Null: sa.types.NullType,
    dt.Date: sa.Date,
    dt.Time: sa.Time,
    dt.Boolean: sa.Boolean,
    dt.Binary: sa.LargeBinary,
    dt.String: sa.Text,
    dt.Decimal: sa.NUMERIC,
    # Mantissa-based
    dt.Float16: sa.REAL,
    dt.Float32: sa.REAL,
    # precision is the number of bits in the mantissa
    # without specifying this, some backends interpret the type as FLOAT, which
    # means float32 (and precision == 24)
    dt.Float64: sa.Float(precision=53),
    dt.Int8: sa.SmallInteger,
    dt.Int16: sa.SmallInteger,
    dt.Int32: sa.Integer,
    dt.Int64: sa.BigInteger,
    dt.UInt8: UInt8,
    dt.UInt16: UInt16,
    dt.UInt32: UInt32,
    dt.UInt64: UInt64,
    dt.JSON: sa.JSON,
    dt.Interval: sa.Interval,
}


_DEFAULT_DIALECT = DefaultDialect()

to_sqla_type = Dispatcher("to_sqla_type")


@to_sqla_type.register(Dialect, dt.DataType)
def _default(_, itype):
    return ibis_type_to_sqla[type(itype)]


@to_sqla_type.register(Dialect, dt.Decimal)
def _decimal(_, itype):
    return sa.types.NUMERIC(itype.precision, itype.scale)


@to_sqla_type.register(Dialect, dt.Timestamp)
def _timestamp(_, itype):
    return sa.TIMESTAMP(timezone=bool(itype.timezone))


@to_sqla_type.register(Dialect, dt.Array)
def _array(dialect, itype):
    return ArrayType(to_sqla_type(dialect, itype.value_type))


@to_sqla_type.register(PGDialect, dt.Array)
def _pg_array(dialect, itype):
    # Unwrap the array element type because sqlalchemy doesn't allow arrays of
    # arrays. This doesn't affect the underlying data.
    while itype.is_array():
        itype = itype.value_type
    return sa.ARRAY(to_sqla_type(dialect, itype))


@to_sqla_type.register(PGDialect, dt.Map)
def _pg_map(dialect, itype):
    if not (itype.key_type.is_string() and itype.value_type.is_string()):
        raise TypeError(f"PostgreSQL only supports map<string, string>, got: {itype}")
    return postgresql.HSTORE


@to_sqla_type.register(Dialect, dt.Struct)
def _struct(dialect, itype):
    return StructType(
        [(name, to_sqla_type(dialect, type)) for name, type in itype.fields.items()]
    )


@to_sqla_type.register(Dialect, dt.Map)
def _map(dialect, itype):
    return MapType(
        to_sqla_type(dialect, itype.key_type), to_sqla_type(dialect, itype.value_type)
    )


@dt.dtype.register(Dialect, sa.types.NullType)
def sa_null(_, satype, nullable=True):
    return dt.null


@dt.dtype.register(Dialect, sa.types.Boolean)
def sa_boolean(_, satype, nullable=True):
    return dt.Boolean(nullable=nullable)


@dt.dtype.register(MySQLDialect, (sa.NUMERIC, mysql.NUMERIC))
def sa_mysql_numeric(_, satype, nullable=True):
    # https://dev.mysql.com/doc/refman/8.0/en/fixed-point-types.html
    return dt.Decimal(satype.precision or 10, satype.scale or 0, nullable=nullable)


_FLOAT_PREC_TO_TYPE = {
    11: dt.Float16,
    24: dt.Float32,
    53: dt.Float64,
}


@dt.dtype.register(Dialect, sa.types.Float)
def sa_float(_, satype, nullable=True):
    precision = satype.precision
    if (typ := _FLOAT_PREC_TO_TYPE.get(precision)) is not None:
        return typ(nullable=nullable)
    return dt.Decimal(precision, satype.scale, nullable=nullable)


@dt.dtype.register(Dialect, sa.types.Numeric)
@dt.dtype.register(SQLiteDialect, sqlite.NUMERIC)
def sa_numeric(_, satype, nullable=True):
    return dt.Decimal(satype.precision, satype.scale, nullable=nullable)


@dt.dtype.register(Dialect, sa.types.SmallInteger)
def sa_smallint(_, satype, nullable=True):
    return dt.Int16(nullable=nullable)


@dt.dtype.register(Dialect, sa.types.Integer)
def sa_integer(_, satype, nullable=True):
    return dt.Int32(nullable=nullable)


@dt.dtype.register(Dialect, mysql.TINYINT)
@dt.dtype.register(MSDialect, mssql.TINYINT)
@dt.dtype.register(MySQLDialect, mysql.YEAR)
def sa_mysql_tinyint(_, satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(MSDialect, mssql.BIT)
def sa_mssql_bit(_, satype, nullable=True):
    return dt.Boolean(nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.BIT)
def sa_mysql_bit(_, satype, nullable=True):
    if 1 <= (length := satype.length) <= 8:
        return dt.Int8(nullable=nullable)
    elif 9 <= length <= 16:
        return dt.Int16(nullable=nullable)
    elif 17 <= length <= 32:
        return dt.Int32(nullable=nullable)
    elif 33 <= length <= 64:
        return dt.Int64(nullable=nullable)
    else:
        raise ValueError(f"Invalid MySQL BIT length: {length:d}")


@dt.dtype.register(Dialect, sa.types.BigInteger)
@dt.dtype.register(MSDialect, mssql.MONEY)
def sa_bigint(_, satype, nullable=True):
    return dt.Int64(nullable=nullable)


@dt.dtype.register(MSDialect, mssql.SMALLMONEY)
def sa_mssql_smallmoney(_, satype, nullable=True):
    return dt.Int32(nullable=nullable)


@dt.dtype.register(Dialect, sa.REAL)
@dt.dtype.register(MySQLDialect, mysql.FLOAT)
def sa_real(_, satype, nullable=True):
    return dt.Float32(nullable=nullable)


@dt.dtype.register(Dialect, sa.FLOAT)
@dt.dtype.register(SQLiteDialect, sa.REAL)
@dt.dtype.register(PGDialect, postgresql.DOUBLE_PRECISION)
def sa_double(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.UUID)
@dt.dtype.register(MSDialect, mssql.UNIQUEIDENTIFIER)
def sa_uuid(_, satype, nullable=True):
    return dt.UUID(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.MACADDR)
def sa_macaddr(_, satype, nullable=True):
    return dt.MACADDR(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.HSTORE)
def sa_hstore(_, satype, nullable=True):
    return dt.Map(dt.string, dt.string, nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.INET)
def sa_inet(_, satype, nullable=True):
    return dt.INET(nullable=nullable)


@dt.dtype.register(Dialect, sa.types.JSON)
@dt.dtype.register(PGDialect, postgresql.JSONB)
def sa_json(_, satype, nullable=True):
    return dt.JSON(nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.TIMESTAMP)
def sa_mysql_timestamp(_, satype, nullable=True):
    return dt.Timestamp(timezone="UTC", nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.DATETIME)
def sa_mysql_datetime(_, satype, nullable=True):
    return dt.Timestamp(nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.SET)
def sa_mysql_set(_, satype, nullable=True):
    return dt.Set(dt.string, nullable=nullable)


if geospatial_supported:

    @dt.dtype.register(Dialect, (ga.Geometry, ga.types._GISType))
    def ga_geometry(_, gatype, nullable=True):
        t = gatype.geometry_type.upper()
        if t == 'POINT':
            return dt.Point(nullable=nullable)
        if t == 'LINESTRING':
            return dt.LineString(nullable=nullable)
        if t == 'POLYGON':
            return dt.Polygon(nullable=nullable)
        if t == 'MULTILINESTRING':
            return dt.MultiLineString(nullable=nullable)
        if t == 'MULTIPOINT':
            return dt.MultiPoint(nullable=nullable)
        if t == 'MULTIPOLYGON':
            return dt.MultiPolygon(nullable=nullable)
        if t in ('GEOMETRY', 'GEOGRAPHY'):
            return getattr(dt, gatype.name.lower())(nullable=nullable)
        else:
            raise ValueError(f"Unrecognized geometry type: {t}")

    @to_sqla_type.register(Dialect, dt.GeoSpatial)
    def _(_, itype, **kwargs):
        if itype.geotype == 'geometry':
            return ga.Geometry
        elif itype.geotype == 'geography':
            return ga.Geography
        else:
            return ga.types._GISType


POSTGRES_FIELD_TO_IBIS_UNIT = {
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


@dt.dtype.register(PGDialect, postgresql.INTERVAL)
def sa_postgres_interval(_, satype, nullable=True):
    field = satype.fields.upper()
    unit = POSTGRES_FIELD_TO_IBIS_UNIT.get(field, None)
    if unit is None:
        raise ValueError(f"Unknown PostgreSQL interval field {field!r}")
    elif unit in {"Y", "M"}:
        raise ValueError(
            "Variable length timedeltas are not yet supported with PostgreSQL"
        )
    return dt.Interval(unit=unit, nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.DOUBLE)
def sa_mysql_double(_, satype, nullable=True):
    # TODO: handle asdecimal=True
    return dt.Float64(nullable=nullable)


@dt.dtype.register(Dialect, sa.types.String)
def sa_string(_, satype, nullable=True):
    return dt.String(nullable=nullable)


@dt.dtype.register(Dialect, sa.LargeBinary)
@dt.dtype.register(MSDialect, (mssql.BINARY, mssql.TIMESTAMP))
@dt.dtype.register(
    MySQLDialect,
    (
        mysql.TINYBLOB,
        mysql.MEDIUMBLOB,
        mysql.BLOB,
        mysql.LONGBLOB,
        mysql.BINARY,
        mysql.VARBINARY,
    ),
)
def sa_binary(_, satype, nullable=True):
    return dt.Binary(nullable=nullable)


@dt.dtype.register(Dialect, sa.Time)
def sa_time(_, satype, nullable=True):
    return dt.Time(nullable=nullable)


@dt.dtype.register(Dialect, sa.Date)
def sa_date(_, satype, nullable=True):
    return dt.Date(nullable=nullable)


@dt.dtype.register(Dialect, sa.DateTime)
def sa_datetime(_, satype, nullable=True, default_timezone='UTC'):
    timezone = default_timezone if satype.timezone else None
    return dt.Timestamp(timezone=timezone, nullable=nullable)


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


@dt.dtype.register(PGDialect, sa.ARRAY)
def sa_pg_array(dialect, satype, nullable=True):
    dimensions = satype.dimensions
    if dimensions is not None and dimensions != 1:
        raise NotImplementedError(
            f"Nested array types not yet supported for {dialect.name} dialect"
        )

    value_dtype = dt.dtype(dialect, satype.item_type)
    return dt.Array(value_dtype, nullable=nullable)


@dt.dtype.register(Dialect, StructType)
def sa_struct(dialect, satype, nullable=True):
    pairs = [(name, dt.dtype(dialect, typ)) for name, typ in satype.pairs]
    return dt.Struct.from_tuples(pairs, nullable=nullable)


@dt.dtype.register(Dialect, ArrayType)
def sa_array(dialect, satype, nullable=True):
    return dt.Array(dt.dtype(dialect, satype.value_type), nullable=nullable)


@sch.infer.register((sa.Table, sa.sql.TableClause))
def schema_from_table(
    table: sa.Table,
    schema: sch.Schema | None = None,
    dialect: sa.engine.interfaces.Dialect | None = None,
) -> sch.Schema:
    """Retrieve an ibis schema from a SQLAlchemy `Table`.

    Parameters
    ----------
    table
        Table whose schema to infer
    schema
        Schema to pull types from
    dialect
        Optional sqlalchemy dialect

    Returns
    -------
    schema
        An ibis schema corresponding to the types of the columns in `table`.
    """
    schema = schema if schema is not None else {}
    pairs = []
    for name, column in zip(table.columns.keys(), table.columns):
        if name in schema:
            dtype = dt.dtype(schema[name])
        else:
            dtype = dt.dtype(
                dialect or getattr(table.bind, "dialect", DefaultDialect()),
                column.type,
                nullable=column.nullable,
            )
        pairs.append((name, dtype))
    return sch.schema(pairs)
