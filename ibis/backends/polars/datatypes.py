from __future__ import annotations

import functools

import polars as pl

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

_to_polars_types = {
    dt.Boolean: pl.Boolean,
    dt.Null: pl.Null,
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

_from_polars_types = {v: k for k, v in _to_polars_types.items()}
_from_polars_types[pl.Categorical] = dt.String


@functools.singledispatch
def dtype_to_polars(dtype):
    """Convert ibis dtype to the polars counterpart."""
    try:
        return _to_polars_types[dtype.__class__]  # else return  pl.Object?
    except KeyError:
        raise NotImplementedError(f"Unsupported type: {dtype!r}")


@dtype_to_polars.register(dt.Decimal)
def from_ibis_decimal(dtype):
    return pl.Decimal(dtype.precision, dtype.scale)


@dtype_to_polars.register(dt.Timestamp)
def from_ibis_timestamp(dtype):
    return pl.Datetime("ns", dtype.timezone)


@dtype_to_polars.register(dt.Interval)
def from_ibis_interval(dtype):
    if dtype.unit.short in {"us", "ns", "ms"}:
        return pl.Duration(dtype.unit.short)
    else:
        raise ValueError(f"Unsupported polars duration unit: {dtype.unit}")


@dtype_to_polars.register(dt.Struct)
def from_ibis_struct(dtype):
    fields = [
        pl.Field(name=name, dtype=dtype_to_polars(dtype))
        for name, dtype in dtype.fields.items()
    ]
    return pl.Struct(fields)


@dtype_to_polars.register(dt.Array)
def from_ibis_array(dtype):
    return pl.List(dtype_to_polars(dtype.value_type))


@functools.singledispatch
def dtype_from_polars(typ):
    """Convert polars dtype to the ibis counterpart."""
    klass = _from_polars_types[typ]
    return klass()


@dtype_from_polars.register(pl.Datetime)
def from_polars_datetime(typ):
    try:
        timezone = typ.time_zone
    except AttributeError:  # pragma: no cover
        timezone = typ.tz  # pragma: no cover
    return dt.Timestamp(timezone=timezone)


@dtype_from_polars.register(pl.Duration)
def from_polars_duration(typ):
    try:
        time_unit = typ.time_unit
    except AttributeError:  # pragma: no cover
        time_unit = typ.tu  # pragma: no cover
    return dt.Interval(unit=time_unit)


@dtype_from_polars.register(pl.List)
def from_polars_list(typ):
    return dt.Array(dtype_from_polars(typ.inner))


@dtype_from_polars.register(pl.Struct)
def from_polars_struct(typ):
    return dt.Struct.from_tuples(
        [(field.name, dtype_from_polars(field.dtype)) for field in typ.fields]
    )


@dtype_from_polars.register(pl.Decimal)
def from_polars_decimal(typ: pl.Decimal):
    return dt.Decimal(precision=typ.precision, scale=typ.scale)


def schema_from_polars(schema: pl.Schema) -> sch.Schema:
    fields = [(name, dtype_from_polars(typ)) for name, typ in schema.items()]
    return sch.Schema.from_tuples(fields)
