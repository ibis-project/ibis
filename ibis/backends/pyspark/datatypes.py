from __future__ import annotations

import pyspark.sql.types as pt

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
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

_pyspark_interval_units = {
    pt.DayTimeIntervalType.SECOND: 's',
    pt.DayTimeIntervalType.MINUTE: 'm',
    pt.DayTimeIntervalType.HOUR: 'h',
    pt.DayTimeIntervalType.DAY: 'D',
}


def dtype_from_pyspark(typ, nullable=True):
    """Convert a pyspark type to an ibis type."""
    if isinstance(typ, pt.DecimalType):
        return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
    elif isinstance(typ, pt.ArrayType):
        return dt.Array(dtype_from_pyspark(typ.elementType), nullable=nullable)
    elif isinstance(typ, pt.MapType):
        return dt.Map(
            dtype_from_pyspark(typ.keyType),
            dtype_from_pyspark(typ.valueType),
            nullable=nullable,
        )
    elif isinstance(typ, pt.StructType):
        fields = {f.name: dtype_from_pyspark(f.dataType) for f in typ.fields}

        return dt.Struct(fields, nullable=nullable)
    elif isinstance(typ, pt.DayTimeIntervalType):
        if typ.startField == typ.endField and typ.startField in _pyspark_interval_units:
            unit = _pyspark_interval_units[typ.startField]
            return dt.Interval(unit, nullable=nullable)
        else:
            raise com.IbisTypeError(f"{typ!r} couldn't be converted to Interval")
    elif isinstance(typ, pt.UserDefinedType):
        return dtype_from_pyspark(typ.sqlType(), nullable=nullable)
    else:
        try:
            return _from_pyspark_dtypes[type(typ)](nullable=nullable)
        except KeyError:
            raise NotImplementedError(
                f'Unable to convert type {typ} of type {type(typ)} to an ibis type.'
            )


def dtype_to_pyspark(dtype):
    if dtype.is_decimal():
        return pt.DecimalType(dtype.precision, dtype.scale)
    elif dtype.is_array():
        element_type = dtype_to_pyspark(dtype.value_type)
        contains_null = dtype.value_type.nullable
        return pt.ArrayType(element_type, contains_null)
    elif dtype.is_map():
        key_type = dtype_to_pyspark(dtype.key_type)
        value_type = dtype_to_pyspark(dtype.value_type)
        value_contains_null = dtype.value_type.nullable
        return pt.MapType(key_type, value_type, value_contains_null)
    elif dtype.is_struct():
        fields = [
            pt.StructField(n, dtype_to_pyspark(t), t.nullable)
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
