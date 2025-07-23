from __future__ import annotations
from ibis.common.collections import FrozenOrderedDict
from ibis.common.temporal import IntervalUnit
import ibis.expr.datatypes.mypy
from collections.abc import Mapping
from typing import TypeAlias, TypeGuard, overload, Union, get_args, Literal

from ibis.expr.datatypes.core import DataType
import pyspark
import pyspark.sql.types as pt
from packaging.version import parse as vparse

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper


# DayTimeIntervalType introduced in Spark 3.2 (at least) but didn't show up in
# PySpark until version 3.3
PYSPARK_33 = vparse(pyspark.__version__) >= vparse("3.3")
PYSPARK_35 = vparse(pyspark.__version__) >= vparse("3.5")

SparkDataType: TypeAlias = Union[
    pt.ArrayType,
    pt.BinaryType,
    pt.BooleanType,
    pt.ByteType,
    pt.DateType,
    pt.DayTimeIntervalType,
    pt.DecimalType,
    pt.DoubleType,
    pt.FloatType,
    pt.IntegerType,
    pt.LongType,
    pt.MapType,
    pt.NullType,
    pt.ShortType,
    pt.StringType,
    pt.StructType,
    pt.TimestampNTZType,
    pt.TimestampType,
    pt.UserDefinedType,
]

def is_supported_spark_type(ty: pt.DataType) -> TypeGuard[SparkDataType]:
    return isinstance(ty, get_args(SparkDataType))


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
_to_pyspark_dtypes[dt.UUID] = pt.StringType


if PYSPARK_33:
    _pyspark_interval_units : dict[int, IntervalUnit] = {
        pt.DayTimeIntervalType.SECOND: IntervalUnit.SECOND,
        pt.DayTimeIntervalType.MINUTE: IntervalUnit.MINUTE,
        pt.DayTimeIntervalType.HOUR: IntervalUnit.HOUR,
        pt.DayTimeIntervalType.DAY: IntervalUnit.DAY,
    }


class PySparkType(TypeMapper):
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.BinaryType, nullable: bool = True) -> dt.Binary: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.BooleanType, nullable: bool = True) -> dt.Boolean: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.ByteType, nullable: bool = True) -> dt.Int8: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.DateType, nullable: bool = True) -> dt.Date: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.DoubleType, nullable: bool = True) -> dt.Float64: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.FloatType, nullable: bool = True) -> dt.Float32: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.IntegerType, nullable: bool = True) -> dt.Int32: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.LongType, nullable: bool = True) -> dt.Int64: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.NullType, nullable: bool = True) -> dt.Null: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.ShortType, nullable: bool = True) -> dt.Int16: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.StringType, nullable: bool = True) -> dt.String: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.TimestampType, nullable: bool = True) -> dt.Timestamp: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.DecimalType, nullable: bool = True) -> dt.Decimal: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.ArrayType, nullable: bool = True) -> dt.Array: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.MapType, nullable: bool = True) -> dt.Map: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.StructType, nullable: bool = True) -> dt.Struct: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.DayTimeIntervalType, nullable: bool = True) -> dt.Interval: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.TimestampNTZType, nullable: bool = True) -> dt.Timestamp: ...
    @classmethod
    @overload
    def to_ibis(cls, typ: pt.UserDefinedType, nullable: bool = True) -> dt.Timestamp: ...

    @classmethod
    def to_ibis(cls, typ: SparkDataType, nullable=True) -> ibis.expr.datatypes.mypy._DataType:
        """Convert a pyspark type to an ibis type."""
        if isinstance(typ, pt.DecimalType):
            return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
        elif isinstance(typ, pt.ArrayType):
            if not is_supported_spark_type(typ.elementType):
                raise NotImplementedError(typ.elementType)
            return dt.Array(value_type=cls.to_ibis(typ.elementType), nullable=nullable)
        elif isinstance(typ, pt.MapType):
            if not is_supported_spark_type(typ.keyType):
                raise NotImplementedError(typ.keyType)
            if not is_supported_spark_type(typ.valueType):
                raise NotImplementedError(typ.valueType)
            return dt.Map(
                key_type=cls.to_ibis(typ.keyType),
                value_type=cls.to_ibis(typ.valueType),
                nullable=nullable,
            )
        elif isinstance(typ, pt.StructType):
            fields: dict[str, ibis.expr.datatypes.mypy._DataType] = {}
            for f in typ.fields:
                if not is_supported_spark_type(f.dataType):
                    raise NotImplementedError(f.dataType)

                fields[f.name] = cls.to_ibis(f.dataType)

            return dt.Struct(fields=fields, nullable=nullable)
        elif PYSPARK_33 and isinstance(typ, pt.DayTimeIntervalType):
            if (
                typ.startField == typ.endField
                and typ.startField in _pyspark_interval_units
            ):
                unit = _pyspark_interval_units[typ.startField]
                return dt.Interval(unit=unit, nullable=nullable)
            else:
                raise com.IbisTypeError(f"{typ!r} couldn't be converted to Interval")
        elif PYSPARK_35 and isinstance(typ, pt.TimestampNTZType):
            return dt.Timestamp(nullable=nullable)
        elif isinstance(typ, pt.UserDefinedType):
            sql_type = typ.sqlType()
            if not is_supported_spark_type(sql_type):
                raise NotImplementedError(sql_type)
            return cls.to_ibis(sql_type, nullable=nullable)
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


class PySparkSchema(SchemaMapper[pt.StructType]):
    @classmethod
    def from_ibis(cls, schema: sch.Schema) -> pt.StructType:
        fields = [
            pt.StructField(name, PySparkType.from_ibis(dtype), dtype.nullable)
            for name, dtype in schema.items()
        ]
        return pt.StructType(fields)

    @classmethod
    def to_ibis(cls, schema: pt.StructType) -> sch.Schema:
        struct = PySparkType.to_ibis(schema)
        return sch.Schema(FrozenOrderedDict(struct.fields))
