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
    dt.Date: pa.date64(),
    dt.Time: pa.time64("us"),
    dt.Timestamp: pa.timestamp('ns'),
    dt.JSON: pa.string(),
    dt.Null: pa.null(),
    # assume unknown types can be converted into strings
    dt.Unknown: pa.string(),
}


@functools.singledispatch
def to_pyarrow_type(dtype: dt.DataType) -> pa.DataType:
    if (arrow_type := _to_pyarrow_types.get(dtype.__class__)) is None:
        raise NotImplementedError(
            f"Unsupported conversion from ibis type to pyarrow type: {dtype!r}"
        )
    return arrow_type


@to_pyarrow_type.register(dt.Array)
def from_ibis_collection(dtype: dt.Array) -> pa.ListType:
    return pa.list_(to_pyarrow_type(dtype.value_type))


@to_pyarrow_type.register
def from_ibis_interval(dtype: dt.Interval) -> pa.DurationType:
    try:
        return pa.duration(dtype.unit.short)
    except ValueError:
        raise com.IbisTypeError(f"Unsupported interval unit: {dtype.unit}")


@to_pyarrow_type.register
def from_ibis_struct(dtype: dt.Struct) -> pa.StructType:
    return pa.struct(
        pa.field(name, to_pyarrow_type(typ)) for name, typ in dtype.fields.items()
    )


@to_pyarrow_type.register
def from_ibis_map(dtype: dt.Map) -> pa.MapType:
    return pa.map_(to_pyarrow_type(dtype.key_type), to_pyarrow_type(dtype.value_type))


@to_pyarrow_type.register
def from_ibis_decimal(dtype: dt.Decimal):
    precision = dtype.precision
    scale = dtype.scale

    # set default precision and scale to something; unclear how to choose this
    if precision is None:
        precision = 38

    if scale is None:
        scale = 9

    if precision <= 38:
        return pa.decimal128(precision, scale)
    elif precision <= 76:
        return pa.decimal256(precision, scale)
    else:
        raise com.IbisError(
            f"Invalid `precision` value for pyarrow decimal types: {precision:d}"
        )


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
    pa.null(): dt.Null,
    pa.time32("s"): dt.Time,
    pa.time32("ms"): dt.Time,
    pa.time64("us"): dt.Time,
    pa.time64("ns"): dt.Time,
}


@dt.dtype.register(pa.DataType)  # type: ignore[misc]
def from_pyarrow_primitive(
    arrow_type: pa.DataType, nullable: bool = True
) -> dt.DataType:
    dtype = _to_ibis_dtypes.get(arrow_type, dt.Unknown)
    return dtype(nullable=nullable)


@dt.dtype.register((pa.Decimal128Type, pa.Decimal256Type))
def from_pyarrow_decimal(
    arrow_type: pa.Decimal128Type | pa.Decimal256Type, nullable: bool = True
) -> dt.Decimal:
    return dt.Decimal(
        precision=arrow_type.precision, scale=arrow_type.scale, nullable=nullable
    )


@dt.dtype.register((pa.Time32Type, pa.Time64Type))  # type: ignore[misc]
def from_pyarrow_time(
    _: pa.Time32Type | pa.Time64Type, nullable: bool = True
) -> dt.Time:
    return dt.Time(nullable=nullable)


@dt.dtype.register(pa.ListType)  # type: ignore[misc]
def from_pyarrow_list(arrow_type: pa.ListType, nullable: bool = True) -> dt.Array:
    return dt.Array(dt.dtype(arrow_type.value_type), nullable=nullable)


@dt.dtype.register(pa.MapType)  # type: ignore[misc]
def from_pyarrow_map(arrow_type: pa.MapType, nullable: bool = True) -> dt.Map:
    return dt.Map(
        dt.dtype(arrow_type.key_type),
        dt.dtype(arrow_type.item_type),
        nullable=nullable,
    )


@dt.dtype.register(pa.StructType)  # type: ignore[misc]
def from_pyarrow_struct(arrow_type: pa.StructType, nullable: bool = True) -> dt.Struct:
    return dt.Struct.from_tuples(
        ((field.name, dt.dtype(field.type)) for field in arrow_type),
        nullable=nullable,
    )


@dt.dtype.register(pa.TimestampType)  # type: ignore[misc]
def from_pyarrow_timestamp(
    arrow_type: pa.TimestampType, nullable: bool = True
) -> dt.Timestamp:
    return dt.Timestamp(timezone=arrow_type.tz, nullable=nullable)


@sch.schema.register(pa.Schema)  # type: ignore[misc]
def from_pyarrow_schema(schema: pa.Schema) -> sch.Schema:
    return sch.schema((f.name, dt.dtype(f.type, nullable=f.nullable)) for f in schema)


def _schema_to_pyarrow_schema_fields(schema: sch.Schema) -> Iterable[pa.Field]:
    return (
        pa.field(name, dtype.to_pyarrow(), nullable=dtype.nullable)
        for name, dtype in schema.items()
    )


def ibis_to_pyarrow_struct(schema: sch.Schema) -> pa.StructType:
    return pa.struct(_schema_to_pyarrow_schema_fields(schema))


def ibis_to_pyarrow_schema(schema: sch.Schema) -> pa.Schema:
    return pa.schema(_schema_to_pyarrow_schema_fields(schema))
