from __future__ import annotations

import sqlalchemy as sa
import sqlalchemy.types as sat
from sqlalchemy.ext.compiler import compiles

import ibis.expr.datatypes as dt
from ibis.backends.base.sqlglot.datatypes import SqlglotType
from ibis.formats import TypeMapper


class UInt64(sat.Integer):
    pass


class UInt32(sat.Integer):
    pass


class UInt16(sat.Integer):
    pass


class UInt8(sat.Integer):
    pass


@compiles(UInt64, "mssql")
@compiles(UInt32, "mssql")
@compiles(UInt16, "mssql")
@compiles(UInt8, "mssql")
@compiles(UInt64, "sqlite")
@compiles(UInt32, "sqlite")
@compiles(UInt16, "sqlite")
@compiles(UInt8, "sqlite")
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
    sat.BLOB: dt.Binary,
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
        elif isinstance(typ, sa.DateTime):
            timezone = "UTC" if typ.timezone else None
            return dt.Timestamp(timezone, nullable=nullable)
        elif isinstance(typ, sat.String):
            return dt.String(nullable=nullable)
        else:
            raise TypeError(f"Unable to convert type: {typ!r}")
