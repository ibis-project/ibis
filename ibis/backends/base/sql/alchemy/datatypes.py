from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
import sqlalchemy.types as sat
import toolz
from sqlalchemy.ext.compiler import compiles

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.backends.base.sqlglot.datatypes import SqlglotType
from ibis.common.collections import FrozenDict
from ibis.formats import TypeMapper

if TYPE_CHECKING:
    from collections.abc import Mapping

if geospatial_supported:
    import geoalchemy2 as ga


class ArrayType(sat.UserDefinedType):
    def __init__(self, value_type: sat.TypeEngine):
        self.value_type = sat.to_instance(value_type)

    def result_processor(self, dialect, coltype) -> None:
        if not coltype.lower().startswith("array"):
            return None

        inner_processor = (
            self.value_type.result_processor(dialect, coltype[len("array(") : -1])
            or toolz.identity
        )

        return lambda v: v if v is None else list(map(inner_processor, v))


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
    quote = compiler.dialect.identifier_preparer.quote
    content = ", ".join(
        f"{quote(field)} {compiler.process(typ, **kw)}"
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
    UUID = sat.UUID
except AttributeError:

    class UUID(sat.String):
        pass

else:

    @compiles(UUID, "default")
    def compiles_uuid(element, compiler, **kw):
        return "UUID"


class Unknown(sa.Text):
    pass


_from_sqlalchemy_types = {
    sat.BOOLEAN: dt.Boolean,
    sat.Boolean: dt.Boolean,
    sat.BINARY: dt.Binary,
    sat.LargeBinary: dt.Binary,
    sat.DATE: dt.Date,
    sat.Date: dt.Date,
    sat.TEXT: dt.String,
    sat.Text: dt.String,
    sat.TIME: dt.Time,
    sat.Time: dt.Time,
    sat.VARCHAR: dt.String,
    sat.CHAR: dt.String,
    sat.String: dt.String,
    sat.SMALLINT: dt.Int16,
    sat.SmallInteger: dt.Int16,
    sat.INTEGER: dt.Int32,
    sat.Integer: dt.Int32,
    sat.BIGINT: dt.Int64,
    sat.BigInteger: dt.Int64,
    sat.REAL: dt.Float32,
    sat.FLOAT: dt.Float64,
    UInt16: dt.UInt16,
    UInt32: dt.UInt32,
    UInt64: dt.UInt64,
    UInt8: dt.UInt8,
    Unknown: dt.Unknown,
    sat.JSON: dt.JSON,
    UUID: dt.UUID,
}

_to_sqlalchemy_types = {
    dt.Null: sat.NullType,
    dt.Date: sat.Date,
    dt.Time: sat.Time,
    dt.Boolean: sat.Boolean,
    dt.Binary: sat.LargeBinary,
    dt.String: sat.Text,
    dt.Decimal: sat.Numeric,
    # Mantissa-based
    dt.Float16: sat.REAL,
    dt.Float32: sat.REAL,
    # precision is the number of bits in the mantissa
    # without specifying this, some backends interpret the type as FLOAT, which
    # means float32 (and precision == 24)
    dt.Float64: sat.FLOAT(precision=53),
    dt.Int8: sat.SmallInteger,
    dt.Int16: sat.SmallInteger,
    dt.Int32: sat.Integer,
    dt.Int64: sat.BigInteger,
    dt.UInt8: UInt8,
    dt.UInt16: UInt16,
    dt.UInt32: UInt32,
    dt.UInt64: UInt64,
    dt.JSON: sat.JSON,
    dt.Interval: sat.Interval,
    dt.Unknown: Unknown,
    dt.MACADDR: sat.Text,
    dt.INET: sat.Text,
    dt.UUID: UUID,
}

_FLOAT_PREC_TO_TYPE = {
    11: dt.Float16,
    24: dt.Float32,
    53: dt.Float64,
}

_GEOSPATIAL_TYPES = {
    "POINT": dt.Point,
    "LINESTRING": dt.LineString,
    "POLYGON": dt.Polygon,
    "MULTILINESTRING": dt.MultiLineString,
    "MULTIPOINT": dt.MultiPoint,
    "MULTIPOLYGON": dt.MultiPolygon,
    "GEOMETRY": dt.Geometry,
    "GEOGRAPHY": dt.Geography,
}


class AlchemyType(TypeMapper):
    @classmethod
    def to_string(cls, dtype: dt.DataType):
        dialect_class = sa.dialects.registry.load(cls.dialect)
        return str(
            sa.types.to_instance(cls.from_ibis(dtype)).compile(dialect=dialect_class())
        )

    @classmethod
    def from_string(cls, type_string, nullable=True):
        return SqlglotType.from_string(type_string, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> sat.TypeEngine:
        """Convert an Ibis type to a SQLAlchemy type.

        Parameters
        ----------
        dtype
            Ibis type to convert.

        Returns
        -------
        SQLAlchemy type.
        """
        if dtype.is_decimal():
            return sat.NUMERIC(dtype.precision, dtype.scale)
        elif dtype.is_timestamp():
            return sat.TIMESTAMP(timezone=bool(dtype.timezone))
        elif dtype.is_array():
            return ArrayType(cls.from_ibis(dtype.value_type))
        elif dtype.is_struct():
            fields = {k: cls.from_ibis(v) for k, v in dtype.fields.items()}
            return StructType(fields)
        elif dtype.is_map():
            return MapType(
                cls.from_ibis(dtype.key_type), cls.from_ibis(dtype.value_type)
            )
        elif dtype.is_geospatial():
            if geospatial_supported:
                if dtype.geotype == "geometry":
                    return ga.Geometry
                elif dtype.geotype == "geography":
                    return ga.Geography
                else:
                    return ga.types._GISType
            else:
                raise TypeError("geospatial types are not supported")
        else:
            return _to_sqlalchemy_types[type(dtype)]

    @classmethod
    def to_ibis(cls, typ: sat.TypeEngine, nullable: bool = True) -> dt.DataType:
        """Convert a SQLAlchemy type to an Ibis type.

        Parameters
        ----------
        typ
            SQLAlchemy type to convert.
        nullable : bool, optional
            Whether the returned type should be nullable.

        Returns
        -------
        Ibis type.
        """
        if dtype := _from_sqlalchemy_types.get(type(typ)):
            return dtype(nullable=nullable)
        elif isinstance(typ, sat.Float):
            if (float_typ := _FLOAT_PREC_TO_TYPE.get(typ.precision)) is not None:
                return float_typ(nullable=nullable)
            return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
        elif isinstance(typ, sat.Numeric):
            return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
        elif isinstance(typ, ArrayType):
            return dt.Array(cls.to_ibis(typ.value_type), nullable=nullable)
        elif isinstance(typ, sat.ARRAY):
            ndim = typ.dimensions
            if ndim is not None and ndim != 1:
                raise NotImplementedError("Nested array types not yet supported")
            return dt.Array(cls.to_ibis(typ.item_type), nullable=nullable)
        elif isinstance(typ, StructType):
            fields = {k: cls.to_ibis(v) for k, v in typ.fields.items()}
            return dt.Struct(fields, nullable=nullable)
        elif isinstance(typ, MapType):
            return dt.Map(
                cls.to_ibis(typ.key_type),
                cls.to_ibis(typ.value_type),
                nullable=nullable,
            )
        elif isinstance(typ, sa.DateTime):
            timezone = "UTC" if typ.timezone else None
            return dt.Timestamp(timezone, nullable=nullable)
        elif isinstance(typ, sat.String):
            return dt.String(nullable=nullable)
        elif geospatial_supported and isinstance(typ, ga.types._GISType):
            name = typ.geometry_type.upper()
            try:
                return _GEOSPATIAL_TYPES[name](geotype=typ.name, nullable=nullable)
            except KeyError:
                raise ValueError(f"Unrecognized geometry type: {name}")
        else:
            raise TypeError(f"Unable to convert type: {typ!r}")
