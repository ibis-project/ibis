import functools

import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

# TODO(kszucs): the following conversions are really rudimentary
# we should have a pyarrow backend which would be responsible
# for conversions between ibis types to pyarrow types

# TODO(kszucs): support nested and parametric types
# consolidate with the logic from the parquet backend


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
}


@dt.dtype.register(pa.DataType)
def from_pyarrow_primitive(arrow_type, nullable=True):
    return _to_ibis_dtypes[arrow_type](nullable=nullable)


@dt.dtype.register(pa.TimestampType)
def from_pyarrow_timestamp(arrow_type, nullable=True):
    return dt.TimestampType(timezone=arrow_type.tz)


@sch.infer.register(pa.Schema)
def infer_pyarrow_schema(schema):
    fields = [(f.name, dt.dtype(f.type, nullable=f.nullable)) for f in schema]
    return sch.schema(fields)


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
}


@functools.singledispatch
def to_pyarrow_type(dtype):
    return _to_pyarrow_types[dtype.__class__]


@to_pyarrow_type.register(dt.Array)
def from_ibis_array(dtype):
    return pa.list_(to_pyarrow_type(dtype.value_type))


@to_pyarrow_type.register(dt.Set)
def from_ibis_set(dtype):
    return pa.list_(to_pyarrow_type(dtype.value_type))


@to_pyarrow_type.register(dt.Interval)
def from_ibis_interval(dtype):
    try:
        return pa.duration(dtype.unit)
    except ValueError:
        raise com.IbisTypeError(f"Unsupported interval unit: {dtype.unit}")
