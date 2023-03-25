from __future__ import annotations

import functools

import pyspark.sql.types as pt

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.registry import sql_type_names

_sql_type_names = dict(sql_type_names, date='date')


def type_to_sql_string(tval):
    if tval.is_decimal():
        return f'decimal({tval.precision}, {tval.scale})'
    name = tval.name.lower()
    try:
        return _sql_type_names[name]
    except KeyError:
        raise com.UnsupportedBackendType(name)


# maps pyspark type class to ibis type class
_SPARK_DTYPE_TO_IBIS_DTYPE = {
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


@dt.dtype.register(pt.DataType)
def _spark_dtype(spark_dtype_obj, nullable=True):
    """Convert Spark SQL type objects to ibis type objects."""
    ibis_type_class = _SPARK_DTYPE_TO_IBIS_DTYPE.get(type(spark_dtype_obj))
    return ibis_type_class(nullable=nullable)


@dt.dtype.register(pt.DecimalType)
def _spark_decimal(spark_dtype_obj, nullable=True):
    precision = spark_dtype_obj.precision
    scale = spark_dtype_obj.scale
    return dt.Decimal(precision, scale, nullable=nullable)


@dt.dtype.register(pt.ArrayType)
def _spark_array(spark_dtype_obj, nullable=True):
    value_type = dt.dtype(
        spark_dtype_obj.elementType, nullable=spark_dtype_obj.containsNull
    )
    return dt.Array(value_type, nullable=nullable)


@dt.dtype.register(pt.MapType)
def _spark_map(spark_dtype_obj, nullable=True):
    key_type = dt.dtype(spark_dtype_obj.keyType)
    value_type = dt.dtype(
        spark_dtype_obj.valueType, nullable=spark_dtype_obj.valueContainsNull
    )
    return dt.Map(key_type, value_type, nullable=nullable)


@dt.dtype.register(pt.StructType)
def _spark_struct(spark_dtype_obj, nullable=True):
    fields = {
        n: dt.dtype(f.dataType, nullable=f.nullable)
        for n, f in zip(spark_dtype_obj.names, spark_dtype_obj.fields)
    }
    return dt.Struct(fields, nullable=nullable)


_SPARK_INTERVAL_TO_IBIS_INTERVAL = {
    pt.DayTimeIntervalType.SECOND: 's',
    pt.DayTimeIntervalType.MINUTE: 'm',
    pt.DayTimeIntervalType.HOUR: 'h',
    pt.DayTimeIntervalType.DAY: 'D',
}


@dt.dtype.register(pt.DayTimeIntervalType)
def _spark_struct(spark_dtype_obj, nullable=True):
    if (
        spark_dtype_obj.startField == spark_dtype_obj.endField
        and spark_dtype_obj.startField in _SPARK_INTERVAL_TO_IBIS_INTERVAL
    ):
        return dt.Interval(
            _SPARK_INTERVAL_TO_IBIS_INTERVAL[spark_dtype_obj.startField],
            nullable=nullable,
        )
    else:
        raise com.IbisTypeError("DayTimeIntervalType couldn't be converted to Interval")


_IBIS_DTYPE_TO_SPARK_DTYPE = {v: k for k, v in _SPARK_DTYPE_TO_IBIS_DTYPE.items()}
_IBIS_DTYPE_TO_SPARK_DTYPE[dt.JSON] = pt.StringType


@functools.singledispatch
def spark_dtype(value, **kwargs):
    raise com.IbisTypeError(f'Value {value!r} is not a valid datatype')


@spark_dtype.register(pt.DataType)
def _spark(value: pt.DataType) -> pt.DataType:
    return value


@spark_dtype.register(dt.DataType)
def _dtype(ibis_dtype_obj):
    """Convert ibis types types to Spark SQL."""
    dtype = _IBIS_DTYPE_TO_SPARK_DTYPE[type(ibis_dtype_obj)]
    return dtype()


@spark_dtype.register(dt.Decimal)
def _decimal(ibis_dtype_obj):
    precision = ibis_dtype_obj.precision
    scale = ibis_dtype_obj.scale
    return pt.DecimalType(precision, scale)


@spark_dtype.register(dt.Array)
def _array(ibis_dtype_obj):
    element_type = spark_dtype(ibis_dtype_obj.value_type)
    contains_null = ibis_dtype_obj.value_type.nullable
    return pt.ArrayType(element_type, contains_null)


@spark_dtype.register(dt.Map)
def _map(ibis_dtype_obj):
    key_type = spark_dtype(ibis_dtype_obj.key_type)
    value_type = spark_dtype(ibis_dtype_obj.value_type)
    value_contains_null = ibis_dtype_obj.value_type.nullable
    return pt.MapType(key_type, value_type, value_contains_null)


@spark_dtype.register(dt.Struct)
def _struct(ibis_dtype_obj):
    fields = [
        pt.StructField(n, spark_dtype(t), t.nullable)
        for n, t in ibis_dtype_obj.fields.items()
    ]
    return pt.StructType(fields)


@spark_dtype.register(sch.Schema)
def _schema(ibis_schem_obj):
    fields = [
        pt.StructField(n, spark_dtype(t), t.nullable) for n, t in ibis_schem_obj.items()
    ]
    return pt.StructType(fields)
