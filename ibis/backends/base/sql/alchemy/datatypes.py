from __future__ import annotations

from typing import Mapping

import sqlalchemy as sa
import sqlalchemy.types as sat
from multipledispatch import Dispatcher
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.default import DefaultDialect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.ext.compiler import compiles

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.common.collections import FrozenDict

if geospatial_supported:
    import geoalchemy2 as ga


class ArrayType(sat.UserDefinedType):
    def __init__(self, value_type: sat.TypeEngine):
        self.value_type = sat.to_instance(value_type)


@compiles(ArrayType, "default")
def compiles_array(element, compiler, **kw):
    return f"ARRAY({compiler.process(element.value_type, **kw)})"


class StructType(sat.UserDefinedType):
    cache_ok = True

    def __init__(self, fields: Mapping[str, sat.TypeEngine]) -> None:
        self.fields = FrozenDict(
            {name: sat.to_instance(typ) for name, typ in fields.items()}
        )


@compiles(StructType, "default")
def compiles_struct(element, compiler, **kw):
    content = ", ".join(
        f"{field} {compiler.process(typ, **kw)}"
        for field, typ in element.fields.items()
    )
    return f"STRUCT({content})"


class MapType(sat.UserDefinedType):
    def __init__(self, key_type: sat.TypeEngine, value_type: sat.TypeEngine):
        self.key_type = sat.to_instance(key_type)
        self.value_type = sat.to_instance(value_type)


@compiles(MapType, "default")
def compiles_map(element, compiler, **kw):
    key_type = compiler.process(element.key_type, **kw)
    value_type = compiler.process(element.value_type, **kw)
    return f"MAP({key_type}, {value_type})"


class UInt64(sat.Integer):
    pass


class UInt32(sat.Integer):
    pass


class UInt16(sat.Integer):
    pass


class UInt8(sat.Integer):
    pass


@compiles(UInt64, "postgresql")
@compiles(UInt32, "postgresql")
@compiles(UInt16, "postgresql")
@compiles(UInt8, "postgresql")
@compiles(UInt64, "mssql")
@compiles(UInt32, "mssql")
@compiles(UInt16, "mssql")
@compiles(UInt8, "mssql")
@compiles(UInt64, "mysql")
@compiles(UInt32, "mysql")
@compiles(UInt16, "mysql")
@compiles(UInt8, "mysql")
@compiles(UInt64, "snowflake")
@compiles(UInt32, "snowflake")
@compiles(UInt16, "snowflake")
@compiles(UInt8, "snowflake")
@compiles(UInt64, "sqlite")
@compiles(UInt32, "sqlite")
@compiles(UInt16, "sqlite")
@compiles(UInt8, "sqlite")
@compiles(UInt64, "trino")
@compiles(UInt32, "trino")
@compiles(UInt16, "trino")
@compiles(UInt8, "trino")
def compile_uint(element, compiler, **kw):
    dialect_name = compiler.dialect.name
    raise TypeError(
        f"unsigned integers are not supported in the {dialect_name} backend"
    )


try:
    UUID = sa.UUID
except AttributeError:
    pass
else:

    @compiles(UUID, "default")
    def compiles_uuid(element, compiler, **kw):
        return "UUID"


class Unknown(sa.Text):
    pass


# TODO(cleanup)
ibis_type_to_sqla = {
    dt.Null: sat.NullType,
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
    dt.Unknown: Unknown,
}


_DEFAULT_DIALECT = DefaultDialect()

to_sqla_type = Dispatcher("to_sqla_type")


@to_sqla_type.register(Dialect, dt.DataType)
def _default(_, itype):
    return ibis_type_to_sqla[type(itype)]


@to_sqla_type.register(Dialect, dt.Decimal)
def _decimal(_, itype):
    return sat.NUMERIC(itype.precision, itype.scale)


@to_sqla_type.register(Dialect, dt.Timestamp)
def _timestamp(_, itype):
    return sa.TIMESTAMP(timezone=bool(itype.timezone))


@to_sqla_type.register(Dialect, dt.Array)
def _array(dialect, itype):
    return ArrayType(to_sqla_type(dialect, itype.value_type))


@to_sqla_type.register(Dialect, dt.Struct)
def _struct(dialect, itype):
    return StructType(
        {name: to_sqla_type(dialect, type) for name, type in itype.fields.items()}
    )


@to_sqla_type.register(Dialect, dt.Map)
def _map(dialect, itype):
    return MapType(
        to_sqla_type(dialect, itype.key_type), to_sqla_type(dialect, itype.value_type)
    )


@to_sqla_type.register(Dialect, dt.UUID)
def _uuid(dialect, itype):
    try:
        return sa.UUID()
    except AttributeError:
        return postgresql.UUID(as_uuid=True)


@dt.dtype.register(Dialect, sat.NullType)
def sa_null(_, satype, nullable=True):
    return dt.null


@dt.dtype.register(Dialect, sat.Boolean)
def sa_boolean(_, satype, nullable=True):
    return dt.Boolean(nullable=nullable)


try:
    UUID = sa.UUID
except AttributeError:
    pass
else:

    @dt.dtype.register(Dialect, UUID)
    def sa_uuid(_, satype, nullable=True):
        return dt.UUID(nullable=nullable)


_FLOAT_PREC_TO_TYPE = {
    11: dt.Float16,
    24: dt.Float32,
    53: dt.Float64,
}


@dt.dtype.register(Dialect, sat.Float)
def sa_float(_, satype, nullable=True):
    precision = satype.precision
    if (typ := _FLOAT_PREC_TO_TYPE.get(precision)) is not None:
        return typ(nullable=nullable)
    return dt.Decimal(precision, satype.scale, nullable=nullable)


@dt.dtype.register(Dialect, sat.Numeric)
def sa_numeric(_, satype, nullable=True):
    return dt.Decimal(satype.precision, satype.scale, nullable=nullable)


@dt.dtype.register(Dialect, sat.SmallInteger)
def sa_smallint(_, satype, nullable=True):
    return dt.Int16(nullable=nullable)


@dt.dtype.register(Dialect, sat.Integer)
def sa_integer(_, satype, nullable=True):
    return dt.Int32(nullable=nullable)


@dt.dtype.register(Dialect, sat.BigInteger)
def sa_bigint(_, satype, nullable=True):
    return dt.Int64(nullable=nullable)


@dt.dtype.register(Dialect, sa.REAL)
def sa_real(_, satype, nullable=True):
    return dt.Float32(nullable=nullable)


@dt.dtype.register(Dialect, sa.FLOAT)
def sa_double(_, satype, nullable=True):
    return dt.Float64(nullable=nullable)


@dt.dtype.register(Dialect, sa.types.JSON)
def sa_json(_, satype, nullable=True):
    return dt.JSON(nullable=nullable)


@dt.dtype.register(Dialect, Unknown)
def sa_unknown(_, satype, nullable=True):
    return dt.Unknown(nullable=nullable)


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


@dt.dtype.register(Dialect, sa.String)
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


@dt.dtype.register(Dialect, StructType)
def sa_struct(dialect, satype, nullable=True):
    return dt.Struct(
        {name: dt.dtype(dialect, typ) for name, typ in satype.fields.items()},
        nullable=nullable,
    )


@dt.dtype.register(Dialect, ArrayType)
def sa_array(dialect, satype, nullable=True):
    return dt.Array(dt.dtype(dialect, satype.value_type), nullable=nullable)


@sch.infer.register(sa.sql.TableClause)
def schema_from_table(
    table: sa.sql.TableClause,
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
    if dialect is None:
        dialect = _DEFAULT_DIALECT
    for column in table.columns:
        name = column.name
        if name in schema:
            dtype = dt.dtype(schema[name])
        else:
            dtype = dt.dtype(dialect, column.type, nullable=column.nullable)
        pairs.append((name, dtype))
    return sch.schema(pairs)
