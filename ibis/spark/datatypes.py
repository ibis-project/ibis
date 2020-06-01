import functools

import pyspark.sql.types as pt

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.expr.schema import Schema

# maps pyspark type class to ibis type class
_SPARK_DTYPE_TO_IBIS_DTYPE = {
    pt.NullType: dt.Null,
    pt.StringType: dt.String,
    pt.BinaryType: dt.Binary,
    pt.BooleanType: dt.Boolean,
    pt.DateType: dt.Date,
    pt.DoubleType: dt.Double,
    pt.FloatType: dt.Float,
    pt.ByteType: dt.Int8,
    pt.IntegerType: dt.Int32,
    pt.LongType: dt.Int64,
    pt.ShortType: dt.Int16,
    pt.TimestampType: dt.Timestamp,
}


@dt.dtype.register(pt.DataType)
def spark_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    """Convert Spark SQL type objects to ibis type objects."""
    ibis_type_class = _SPARK_DTYPE_TO_IBIS_DTYPE.get(type(spark_dtype_obj))
    return ibis_type_class(nullable=nullable)


@dt.dtype.register(pt.DecimalType)
def spark_decimal_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    precision = spark_dtype_obj.precision
    scale = spark_dtype_obj.scale
    return dt.Decimal(precision, scale, nullable=nullable)


@dt.dtype.register(pt.ArrayType)
def spark_array_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    value_type = dt.dtype(
        spark_dtype_obj.elementType, nullable=spark_dtype_obj.containsNull
    )
    return dt.Array(value_type, nullable=nullable)


@dt.dtype.register(pt.MapType)
def spark_map_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    key_type = dt.dtype(spark_dtype_obj.keyType)
    value_type = dt.dtype(
        spark_dtype_obj.valueType, nullable=spark_dtype_obj.valueContainsNull
    )
    return dt.Map(key_type, value_type, nullable=nullable)


@dt.dtype.register(pt.StructType)
def spark_struct_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    names = spark_dtype_obj.names
    fields = spark_dtype_obj.fields
    ibis_types = [dt.dtype(f.dataType, nullable=f.nullable) for f in fields]
    return dt.Struct(names, ibis_types, nullable=nullable)


_IBIS_DTYPE_TO_SPARK_DTYPE = {
    v: k for k, v in _SPARK_DTYPE_TO_IBIS_DTYPE.items()
}

spark_dtype = functools.singledispatch('spark_dtype')
# from multipledispatch import Dispatcher
# spark_dtype = Dispatcher('spark_dtype')


@spark_dtype.register(object)
def default(value, **kwargs) -> pt.DataType:
    raise com.IbisTypeError('Value {!r} is not a valid datatype'.format(value))


@spark_dtype.register(pt.DataType)
def from_spark_dtype(value: pt.DataType) -> pt.DataType:
    return value


@spark_dtype.register(dt.DataType)
def ibis_dtype_to_spark_dtype(ibis_dtype_obj):
    """Convert ibis types types to Spark SQL."""
    return _IBIS_DTYPE_TO_SPARK_DTYPE.get(type(ibis_dtype_obj))()


@spark_dtype.register(dt.Decimal)
def ibis_decimal_dtype_to_spark_dtype(ibis_dtype_obj):
    precision = ibis_dtype_obj.precision
    scale = ibis_dtype_obj.scale
    return pt.DecimalType(precision, scale)


@spark_dtype.register(dt.Array)
def ibis_array_dtype_to_spark_dtype(ibis_dtype_obj):
    element_type = spark_dtype(ibis_dtype_obj.value_type)
    contains_null = ibis_dtype_obj.value_type.nullable
    return pt.ArrayType(element_type, contains_null)


@spark_dtype.register(dt.Map)
def ibis_map_dtype_to_spark_dtype(ibis_dtype_obj):
    key_type = spark_dtype(ibis_dtype_obj.key_type)
    value_type = spark_dtype(ibis_dtype_obj.value_type)
    value_contains_null = ibis_dtype_obj.value_type.nullable
    return pt.MapType(key_type, value_type, value_contains_null)


@spark_dtype.register(dt.Struct)
@spark_dtype.register(Schema)
def ibis_struct_dtype_to_spark_dtype(ibis_dtype_obj):
    fields = [
        pt.StructField(n, spark_dtype(t), t.nullable)
        for n, t in zip(ibis_dtype_obj.names, ibis_dtype_obj.types)
    ]
    return pt.StructType(fields)
