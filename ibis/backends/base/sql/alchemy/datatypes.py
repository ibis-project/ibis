from __future__ import annotations

import functools
from typing import Iterable

import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.dialects.mysql.base import MySQLDialect
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import UserDefinedType

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported

if geospatial_supported:
    import geoalchemy2 as ga


class StructType(UserDefinedType):
    def __init__(
        self,
        pairs: Iterable[tuple[str, sa.types.TypeEngine]],
    ):
        self.pairs = [
            (name, sa.types.to_instance(type)) for name, type in pairs
        ]

    def get_col_spec(self, **_):
        pairs = ", ".join(f"{k} {v}" for k, v in self.pairs)
        return f"STRUCT({pairs})"


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

    for colname, dtype in zip(schema.names, schema.types):
        satype = to_sqla_type(dtype)
        column = sa.Column(colname, satype, nullable=dtype.nullable)
        columns.append(column)

    return sa.Table(name, meta, schema=database, *columns)


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
}


@functools.singledispatch
def to_sqla_type(itype, type_map=None):
    if type_map is None:
        type_map = ibis_type_to_sqla
    return type_map[type(itype)]


@to_sqla_type.register(dt.Decimal)
def _(itype, **kwargs):
    return sa.types.NUMERIC(itype.precision, itype.scale)


@to_sqla_type.register(dt.Interval)
def _(itype, **kwargs):
    return sa.types.Interval()


@to_sqla_type.register(dt.Date)
def _(itype, **kwargs):
    return sa.Date()


@to_sqla_type.register(dt.Timestamp)
def _(itype, **kwargs):
    return sa.TIMESTAMP(bool(itype.timezone))


@to_sqla_type.register(dt.Array)
def _(itype, **kwargs):
    # Unwrap the array element type because sqlalchemy doesn't allow arrays of
    # arrays. This doesn't affect the underlying data.
    while isinstance(itype, dt.Array):
        itype = itype.value_type
    return sa.ARRAY(to_sqla_type(itype, **kwargs))


@to_sqla_type.register(dt.Struct)
def _(itype, **_):
    return StructType(
        [(name, to_sqla_type(type)) for name, type in itype.pairs.items()]
    )


@to_sqla_type.register(dt.GeoSpatial)
def _(itype, **kwargs):
    if itype.geotype == 'geometry':
        return ga.Geometry
    elif itype.geotype == 'geography':
        return ga.Geography
    else:
        return ga.types._GISType


@dt.dtype.register(Dialect, sa.types.NullType)
def sa_null(_, satype, nullable=True):
    return dt.null


@dt.dtype.register(Dialect, sa.types.Boolean)
def sa_boolean(_, satype, nullable=True):
    return dt.Boolean(nullable=nullable)


@dt.dtype.register(MySQLDialect, mysql.NUMERIC)
def sa_mysql_numeric(_, satype, nullable=True):
    # https://dev.mysql.com/doc/refman/8.0/en/fixed-point-types.html
    return dt.Decimal(
        satype.precision or 10, satype.scale or 0, nullable=nullable
    )


@dt.dtype.register(MySQLDialect, mysql.TINYBLOB)
@dt.dtype.register(MySQLDialect, mysql.MEDIUMBLOB)
@dt.dtype.register(MySQLDialect, mysql.BLOB)
@dt.dtype.register(MySQLDialect, mysql.LONGBLOB)
def sa_mysql_blob(_, satype, nullable=True):
    return dt.Binary(nullable=nullable)


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
def sa_mysql_tinyint(_, satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(Dialect, sa.types.BigInteger)
def sa_bigint(_, satype, nullable=True):
    return dt.Int64(nullable=nullable)


@dt.dtype.register(Dialect, sa.REAL)
def sa_real(_, satype, nullable=True):
    return dt.Float32(nullable=nullable)


@dt.dtype.register(Dialect, sa.FLOAT)
@dt.dtype.register(SQLiteDialect, sa.REAL)
@dt.dtype.register(PGDialect, postgresql.DOUBLE_PRECISION)
def sa_double(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.UUID)
def sa_uuid(_, satype, nullable=True):
    return dt.UUID(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.MACADDR)
def sa_macaddr(_, satype, nullable=True):
    return dt.MACADDR(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.INET)
def sa_inet(_, satype, nullable=True):
    return dt.INET(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.JSON)
def sa_json(_, satype, nullable=True):
    return dt.JSON(nullable=nullable)


@dt.dtype.register(PGDialect, postgresql.JSONB)
def sa_jsonb(_, satype, nullable=True):
    return dt.JSONB(nullable=nullable)


if geospatial_supported:

    @dt.dtype.register(Dialect, (ga.Geometry, ga.types._GISType))
    def ga_geometry(_, gatype, nullable=True):
        t = gatype.geometry_type
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
        if t == 'GEOMETRY':
            return dt.Geometry(nullable=nullable)
        else:
            raise ValueError(f"Unrecognized geometry type: {t}")


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


@dt.dtype.register(Dialect, sa.ARRAY)
def sa_array(dialect, satype, nullable=True):
    dimensions = satype.dimensions
    if dimensions is not None and dimensions != 1:
        raise NotImplementedError('Nested array types not yet supported')

    value_dtype = dt.dtype(dialect, satype.item_type)
    return dt.Array(value_dtype, nullable=nullable)


@dt.dtype.register(Dialect, StructType)
def sa_struct(dialect, satype, nullable=True):
    pairs = [(name, dt.dtype(dialect, typ)) for name, typ in satype.pairs]
    return dt.Struct.from_tuples(pairs, nullable=nullable)


@sch.infer.register((sa.Table, sa.sql.TableClause))
def schema_from_table(table, schema=None):
    """Retrieve an ibis schema from a SQLAlchemy ``Table``.

    Parameters
    ----------
    table : sa.Table

    Returns
    -------
    schema : ibis.expr.datatypes.Schema
        An ibis schema corresponding to the types of the columns in `table`.
    """
    schema = schema if schema is not None else {}
    pairs = []
    for name, column in zip(table.columns.keys(), table.columns):
        if name in schema:
            dtype = dt.dtype(schema[name])
        else:
            dtype = dt.dtype(
                getattr(table.bind, 'dialect', Dialect()),
                column.type,
                nullable=column.nullable,
            )
        pairs.append((name, dtype))
    return sch.schema(pairs)
