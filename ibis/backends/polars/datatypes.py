from __future__ import annotations

import functools

import polars as pl

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

_to_polars_types = {
    dt.Boolean: pl.Boolean,
    dt.Null: pl.Null,
    dt.Array: pl.List,
    dt.String: pl.Utf8,
    dt.Binary: pl.Binary,
    dt.Date: pl.Date,
    dt.Time: pl.Time,
    dt.Int8: pl.Int8,
    dt.Int16: pl.Int16,
    dt.Int32: pl.Int32,
    dt.Int64: pl.Int64,
    dt.UInt8: pl.UInt8,
    dt.UInt16: pl.UInt16,
    dt.UInt32: pl.UInt32,
    dt.UInt64: pl.UInt64,
    dt.Float32: pl.Float32,
    dt.Float64: pl.Float64,
}

_to_ibis_dtypes = {v: k for k, v in _to_polars_types.items()}


@functools.singledispatch
def to_polars_type(dtype):
    """Convert ibis dtype to the polars counterpart."""
    try:
        return _to_polars_types[dtype.__class__]  # else return  pl.Object?
    except KeyError:
        raise NotImplementedError(
            f"Translation to polars dtype not implemented for {dtype}"
        )


@to_polars_type.register(dt.Timestamp)
def from_ibis_timestamp(dtype):
    return pl.Datetime("ns", dtype.timezone)


@to_polars_type.register(dt.Interval)
def from_ibis_interval(dtype):
    if dtype.unit in {'us', 'ns', 'ms'}:
        return pl.Duration(dtype.unit)
    else:
        raise ValueError(f"Unsupported polars duration unit: {dtype.unit}")


@to_polars_type.register(dt.Struct)
def from_ibis_struct(dtype):
    fields = [
        pl.Field(name=name, dtype=to_polars_type(dtype))
        for name, dtype in dtype.fields.items()
    ]
    return pl.Struct(fields)


@to_polars_type.register(dt.Category)
def from_ibis_category(dtype):
    return pl.Categorical


@to_polars_type.register(dt.Array)
def from_ibis_array(dtype):
    return pl.List(to_polars_type(dtype.value_type))


@functools.singledispatch
def to_ibis_dtype(typ):
    """Convert polars dtype to the ibis counterpart."""
    klass = _to_ibis_dtypes[typ]
    return klass()


@to_ibis_dtype.register(pl.Datetime)
def from_polars_datetime(typ):
    # TODO(kszucs): handle timezone?
    return dt.Timestamp()


@to_ibis_dtype.register(pl.List)
def from_polars_list(typ):
    return dt.Array(to_ibis_dtype(typ.inner))


@to_ibis_dtype.register(pl.Struct)
def from_polars_struct(typ):
    return dt.Struct.from_tuples(
        [(field.name, to_ibis_dtype(field.dtype)) for field in typ.fields]
    )


# Can't register here since polars datatypes are classes
# @dt.dtype.register(pl.DataType)


@sch.infer.register(pl.LazyFrame)
def from_polars_schema(df: pl.LazyFrame) -> sch.Schema:
    fields = [(name, to_ibis_dtype(typ)) for name, typ in df.schema.items()]
    return sch.Schema.from_tuples(fields)
