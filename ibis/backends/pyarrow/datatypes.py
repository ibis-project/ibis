from __future__ import annotations

import functools
from typing import Iterable

import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

_to_pyarrow_types = {
    dt.Int8: pa.int8(),
    dt.Int16: pa.int16(),
    dt.Int32: pa.int32(),
    dt.Int64: pa.int64(),
    dt.UInt8: pa.uint8(),
    dt.UInt16: pa.uint16(),
    dt.UInt32: pa.uint32(),
    dt.UInt64: pa.uint64(),
    dt.Float16: pa.float16(),
    dt.Float32: pa.float32(),
    dt.Float64: pa.float64(),
    dt.String: pa.string(),
    dt.Binary: pa.binary(),
    dt.Boolean: pa.bool_(),
    dt.Timestamp: pa.timestamp('ns'),
    dt.Date: pa.date64(),
}


@functools.singledispatch
def to_pyarrow_type(dtype: dt.DataType):
    return _to_pyarrow_types[dtype.__class__]


@to_pyarrow_type.register(dt.Array)
@to_pyarrow_type.register(dt.Set)
def from_ibis_collection(dtype: dt.Array | dt.Set):
    return pa.list_(to_pyarrow_type(dtype.value_type))


@to_pyarrow_type.register
def from_ibis_interval(dtype: dt.Interval):
    try:
        return pa.duration(dtype.unit)
    except ValueError:
        raise com.IbisTypeError(f"Unsupported interval unit: {dtype.unit}")


@to_pyarrow_type.register
def from_ibis_struct(dtype: dt.Struct):
    return pa.struct(
        pa.field(name, to_pyarrow_type(typ)) for name, typ in dtype.fields.items()
    )


@to_pyarrow_type.register
def from_ibis_map(dtype: dt.Map):
    return pa.map_(to_pyarrow_type(dtype.key_type), to_pyarrow_type(dtype.value_type))


_to_ibis_dtypes = {
    pa.int8(): dt.Int8,
    pa.int16(): dt.Int16,
    pa.int32(): dt.Int32,
    pa.int64(): dt.Int64,
    pa.uint8(): dt.UInt8,
    pa.uint16(): dt.UInt16,
    pa.uint32(): dt.UInt32,
    pa.uint64(): dt.UInt64,
    pa.float16(): dt.Float16,
    pa.float32(): dt.Float32,
    pa.float64(): dt.Float64,
    pa.string(): dt.String,
    pa.binary(): dt.Binary,
    pa.bool_(): dt.Boolean,
    pa.date32(): dt.Date,
    pa.date64(): dt.Date,
}


@dt.dtype.register(pa.DataType)  # type: ignore[misc]
def from_pyarrow_primitive(
    arrow_type: pa.DataType,
    nullable: bool = True,
) -> dt.DataType:
    return _to_ibis_dtypes[arrow_type](nullable=nullable)


@dt.dtype.register(pa.Time32Type)  # type: ignore[misc]
@dt.dtype.register(pa.Time64Type)  # type: ignore[misc]
def from_pyarrow_time(
    arrow_type: pa.TimestampType,
    nullable: bool = True,
) -> dt.DataType:
    return dt.Time(nullable=nullable)


@dt.dtype.register(pa.ListType)  # type: ignore[misc]
def from_pyarrow_list(arrow_type: pa.ListType, nullable: bool = True) -> dt.DataType:
    return dt.Array(dt.dtype(arrow_type.value_type), nullable=nullable)


@dt.dtype.register(pa.MapType)  # type: ignore[misc]
def from_pyarrow_map(arrow_type: pa.MapType, nullable: bool = True) -> dt.DataType:
    return dt.Map(
        dt.dtype(arrow_type.key_type),
        dt.dtype(arrow_type.item_type),
        nullable=nullable,
    )


@dt.dtype.register(pa.StructType)  # type: ignore[misc]
def from_pyarrow_struct(
    arrow_type: pa.StructType,
    nullable: bool = True,
) -> dt.DataType:
    return dt.Struct.from_tuples(
        ((field.name, dt.dtype(field.type)) for field in arrow_type),
        nullable=nullable,
    )


@dt.dtype.register(pa.TimestampType)  # type: ignore[misc]
def from_pyarrow_timestamp(
    arrow_type: pa.TimestampType,
    nullable: bool = True,
) -> dt.DataType:
    return dt.Timestamp(timezone=arrow_type.tz)


@sch.schema.register(pa.Schema)  # type: ignore[misc]
def from_pyarrow_schema(schema: pa.Schema) -> sch.Schema:
    return sch.schema([(f.name, dt.dtype(f.type, nullable=f.nullable)) for f in schema])


def _schema_to_pyarrow_schema_fields(schema: sch.Schema) -> Iterable[pa.Field]:
    for name, dtype in schema.items():
        yield pa.field(name, dtype.to_pyarrow(), nullable=dtype.nullable)


def ibis_to_pyarrow_struct(schema: sch.Schema) -> pa.StructType:
    return pa.struct(_schema_to_pyarrow_schema_fields(schema))


def ibis_to_pyarrow_schema(schema: sch.Schema) -> pa.Schema:
    return pa.schema(_schema_to_pyarrow_schema_fields(schema))
