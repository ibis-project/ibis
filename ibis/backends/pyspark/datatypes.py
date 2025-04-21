from __future__ import annotations

from functools import partial
from inspect import isclass

import pyspark.sql.types as pt

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper

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
}

try:
    _from_pyspark_dtypes[pt.TimestampNTZType] = dt.Timestamp
except AttributeError:
    _from_pyspark_dtypes[pt.TimestampType] = dt.Timestamp
else:
    _from_pyspark_dtypes[pt.TimestampType] = partial(dt.Timestamp, timezone="UTC")

_to_pyspark_dtypes = {
    v: k
    for k, v in _from_pyspark_dtypes.items()
    if isclass(v) and not issubclass(v, dt.Timestamp) and not isinstance(v, partial)
}
_to_pyspark_dtypes[dt.JSON] = pt.StringType
_to_pyspark_dtypes[dt.UUID] = pt.StringType


class PySparkType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ, nullable=True):
        """Convert a pyspark type to an ibis type."""
        from ibis.backends.pyspark import SUPPORTS_TIMESTAMP_NTZ

        if isinstance(typ, pt.DecimalType):
            return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
        elif isinstance(typ, pt.ArrayType):
            return dt.Array(cls.to_ibis(typ.elementType), nullable=nullable)
        elif isinstance(typ, pt.MapType):
            return dt.Map(
                cls.to_ibis(typ.keyType), cls.to_ibis(typ.valueType), nullable=nullable
            )
        elif isinstance(typ, pt.StructType):
            fields = {f.name: cls.to_ibis(f.dataType) for f in typ.fields}

            return dt.Struct(fields, nullable=nullable)
        elif isinstance(typ, pt.DayTimeIntervalType):
            pyspark_interval_units = {
                pt.DayTimeIntervalType.SECOND: "s",
                pt.DayTimeIntervalType.MINUTE: "m",
                pt.DayTimeIntervalType.HOUR: "h",
                pt.DayTimeIntervalType.DAY: "D",
            }

            if (
                typ.startField == typ.endField
                and typ.startField in pyspark_interval_units
            ):
                unit = pyspark_interval_units[typ.startField]
                return dt.Interval(unit, nullable=nullable)
            else:
                raise com.IbisTypeError(f"{typ!r} couldn't be converted to Interval")
        elif isinstance(typ, pt.TimestampNTZType):
            if SUPPORTS_TIMESTAMP_NTZ:
                return dt.Timestamp(nullable=nullable)
            raise com.UnsupportedBackendType(
                "PySpark<3.4 doesn't properly support timestamps without a timezone"
            )
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
        from ibis.backends.pyspark import SUPPORTS_TIMESTAMP_NTZ

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
            return pt.StructType(
                [
                    pt.StructField(field, cls.from_ibis(dtype), dtype.nullable)
                    for field, dtype in dtype.fields.items()
                ]
            )
        elif dtype.is_timestamp():
            if dtype.timezone is not None:
                return pt.TimestampType()
            else:
                if not SUPPORTS_TIMESTAMP_NTZ:
                    raise com.UnsupportedBackendType(
                        "PySpark<3.4 doesn't properly support timestamps without a timezone"
                    )
                return pt.TimestampNTZType()
        else:
            try:
                return _to_pyspark_dtypes[type(dtype)]()
            except KeyError:
                raise com.IbisTypeError(
                    f"Unable to convert dtype {dtype!r} to pyspark type"
                )


class PySparkSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema):
        return PySparkType.from_ibis(schema.as_struct())

    @classmethod
    def to_ibis(cls, schema):
        return sch.Schema({name: PySparkType.to_ibis(typ) for name, typ in schema})
