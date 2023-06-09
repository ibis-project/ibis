from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

import ibis.expr.datatypes as dt
from ibis.expr.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Sequence

_from_pyarrow_types = {
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
    pa.string(): dt.String,
    pa.large_binary(): dt.Binary,
    pa.large_string(): dt.String,
    pa.binary(): dt.Binary,
}

_to_pyarrow_types = {
    dt.Null: pa.null(),
    dt.Boolean: pa.bool_(),
    dt.Binary: pa.binary(),
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
    # assume unknown types can be converted into strings
    dt.Unknown: pa.string(),
    dt.MACADDR: pa.string(),
    dt.INET: pa.string(),
}


def dtype_from_pyarrow(typ: pa.DataType, nullable=True) -> dt.DataType:
    """Convert a pyarrow type to an ibis type."""

    if pa.types.is_null(typ):
        return dt.null
    elif pa.types.is_decimal(typ):
        return dt.Decimal(typ.precision, typ.scale, nullable=nullable)
    elif pa.types.is_timestamp(typ):
        return dt.Timestamp.from_unit(typ.unit, timezone=typ.tz, nullable=nullable)
    elif pa.types.is_time(typ):
        return dt.Time(nullable=nullable)
    elif pa.types.is_duration(typ):
        return dt.Interval(typ.unit, nullable=nullable)
    elif pa.types.is_interval(typ):
        raise ValueError("Arrow interval type is not supported")
    elif (
        pa.types.is_list(typ)
        or pa.types.is_large_list(typ)
        or pa.types.is_fixed_size_list(typ)
    ):
        value_dtype = dtype_from_pyarrow(typ.value_type, typ.value_field.nullable)
        return dt.Array(value_dtype, nullable=nullable)
    elif pa.types.is_struct(typ):
        field_dtypes = {
            field.name: dtype_from_pyarrow(field.type, field.nullable) for field in typ
        }
        return dt.Struct(field_dtypes, nullable=nullable)
    elif pa.types.is_map(typ):
        # TODO(kszucs): keys_sorted has just been exposed in pyarrow
        key_dtype = dtype_from_pyarrow(typ.key_type, typ.key_field.nullable)
        value_dtype = dtype_from_pyarrow(typ.item_type, typ.item_field.nullable)
        return dt.Map(key_dtype, value_dtype, nullable=nullable)
    else:
        return _from_pyarrow_types[typ](nullable=nullable)


def dtype_to_pyarrow(dtype: dt.DataType) -> pa.DataType:
    if dtype.is_decimal():
        # set default precision and scale to something; unclear how to choose this
        precision = 38 if dtype.precision is None else dtype.precision
        scale = 9 if dtype.scale is None else dtype.scale

        if precision > 76:
            raise ValueError(
                f"Unsupported precision {dtype.precision} for decimal type"
            )
        elif precision > 38:
            return pa.decimal256(precision, scale)
        else:
            return pa.decimal128(precision, scale)
    elif dtype.is_timestamp():
        return pa.timestamp(
            dtype.unit.short if dtype.scale is not None else "us", tz=dtype.timezone
        )
    elif dtype.is_interval():
        return pa.duration(dtype.unit.short)
    elif dtype.is_time():
        return pa.time64("ns")
    elif dtype.is_date():
        return pa.date64()
    elif dtype.is_array():
        value_field = pa.field(
            'item',
            dtype_to_pyarrow(dtype.value_type),
            nullable=dtype.value_type.nullable,
        )
        return pa.list_(value_field)
    elif dtype.is_struct():
        fields = [
            pa.field(name, dtype_to_pyarrow(dtype), nullable=dtype.nullable)
            for name, dtype in dtype.items()
        ]
        return pa.struct(fields)
    elif dtype.is_map():
        key_field = pa.field(
            'key', dtype_to_pyarrow(dtype.key_type), nullable=dtype.key_type.nullable
        )
        value_field = pa.field(
            'value',
            dtype_to_pyarrow(dtype.value_type),
            nullable=dtype.value_type.nullable,
        )
        return pa.map_(key_field, value_field, keys_sorted=False)
    else:
        try:
            return _to_pyarrow_types[type(dtype)]
        except KeyError:
            raise NotImplementedError(
                f"Converting {dtype} to pyarrow is not supported yet"
            )


def schema_from_pyarrow(schema: pa.Schema) -> Schema:
    fields = [(f.name, dtype_from_pyarrow(f.type, f.nullable)) for f in schema]
    return Schema.from_tuples(fields)


def schema_to_pyarrow(schema: Schema) -> pa.Schema:
    fields = [
        pa.field(name, dtype_to_pyarrow(dtype), nullable=dtype.nullable)
        for name, dtype in schema.items()
    ]
    return pa.schema(fields)


def infer_sequence_dtype(sequence: Sequence) -> dt.DataType:
    try:
        pyarrow_type = pa.array(sequence, from_pandas=True).type
        # pyarrow_type = pa.infer_type(sequence, from_pandas=True)
    except pa.ArrowInvalid:
        try:
            # handle embedded series objects
            return dt.highest_precedence(map(dt.infer, sequence))
        except TypeError:
            # we can still have a type error, e.g., float64 and string in the
            # same array
            return dt.unknown
    except pa.ArrowTypeError:
        # arrow can't infer the type
        return dt.unknown
    else:
        # arrow inferred the type, now convert that type to an ibis type
        return dtype_from_pyarrow(pyarrow_type)
