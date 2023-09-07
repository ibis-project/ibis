from __future__ import annotations

import pyspark
import pyspark.sql.types as pt
from packaging.version import parse as vparse

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends.base.sql.registry import sql_type_names
from ibis.formats import TypeMapper

_sql_type_names = dict(sql_type_names, date="date")

# DayTimeIntervalType introduced in Spark 3.2 (at least) but didn't show up in
# PySpark until version 3.3
PYSPARK_33 = vparse(pyspark.__version__) >= vparse("3.3")


def type_to_sql_string(tval):
    if tval.is_decimal():
        return f"decimal({tval.precision}, {tval.scale})"
    name = tval.name.lower()
    try:
        return _sql_type_names[name]
    except KeyError:
        raise com.UnsupportedBackendType(name)


_from_pyspark_dtypes = {
    pt.BinaryType: dt.Binary,
    pt.BooleanType: dt.Boolean,
    pt.ByteType: dt.Int8,
    pt.DateType: dt.Date,
    pt.DoubleType: dt.Float64,
    pt.FloatType: dt.Float32,
    pt.IntegerType: dt.Int32,
    pt.LongType: dt.Int64,
    pt.NullType: dt.Null,
    pt.ShortType: dt.Int16,
    pt.StringType: dt.String,
    pt.TimestampType: dt.Timestamp,
}

_to_pyspark_dtypes = {v: k for k, v in _from_pyspark_dtypes.items()}
_to_pyspark_dtypes[dt.JSON] = pt.StringType

if PYSPARK_33:
    _pyspark_interval_units = {
        pt.DayTimeIntervalType.SECOND: "s",
        pt.DayTimeIntervalType.MINUTE: "m",
        pt.DayTimeIntervalType.HOUR: "h",
        pt.DayTimeIntervalType.DAY: "D",
    }


class PySparkType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ, nullable=True):
        """Convert a pyspark type to an ibis type."""
        if isinstance(typ, pt.DecimalType):
            return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
        elif isinstance(typ, pt.ArrayType):
            return dt.Array(cls.to_ibis(typ.elementType), nullable=nullable)
        elif isinstance(typ, pt.MapType):
            return dt.Map(
                cls.to_ibis(typ.keyType),
                cls.to_ibis(typ.valueType),
                nullable=nullable,
            )
        elif isinstance(typ, pt.StructType):
            fields = {f.name: cls.to_ibis(f.dataType) for f in typ.fields}

            return dt.Struct(fields, nullable=nullable)
        elif PYSPARK_33 and isinstance(typ, pt.DayTimeIntervalType):
            if (
                typ.startField == typ.endField
                and typ.startField in _pyspark_interval_units
            ):
                unit = _pyspark_interval_units[typ.startField]
                return dt.Interval(unit, nullable=nullable)
            else:
                raise com.IbisTypeError(f"{typ!r} couldn't be converted to Interval")
        elif isinstance(typ, pt.UserDefinedType):
            return cls.to_ibis(typ.sqlType(), nullable=nullable)
        else:
            try:
                return _from_pyspark_dtypes[type(typ)](nullable=nullable)
            except KeyError:
                raise NotImplementedError(
                    f"Unable to convert type {typ} of type {type(typ)} to an ibis type."
                )

    @classmethod
    def from_ibis(cls, dtype):
        if dtype.is_decimal():
            return pt.DecimalType(dtype.precision, dtype.scale)
        elif dtype.is_array():
            element_type = cls.from_ibis(dtype.value_type)
            contains_null = dtype.value_type.nullable
            return pt.ArrayType(element_type, contains_null)
        elif dtype.is_map():
            key_type = cls.from_ibis(dtype.key_type)
            value_type = cls.from_ibis(dtype.value_type)
            value_contains_null = dtype.value_type.nullable
            return pt.MapType(key_type, value_type, value_contains_null)
        elif dtype.is_struct():
            fields = [
                pt.StructField(n, cls.from_ibis(t), t.nullable)
                for n, t in dtype.fields.items()
            ]
            return pt.StructType(fields)
        else:
            try:
                return _to_pyspark_dtypes[type(dtype)]()
            except KeyError:
                raise com.IbisTypeError(
                    f"Unable to convert dtype {dtype!r} to pyspark type"
                )
