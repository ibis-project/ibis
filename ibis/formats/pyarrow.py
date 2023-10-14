from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

import ibis.expr.datatypes as dt
from ibis.expr.schema import Schema
from ibis.formats import DataMapper, SchemaMapper, TypeMapper

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
    dt.JSON: pa.string(),
}


class PyArrowType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: pa.DataType, nullable=True) -> dt.DataType:
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
            value_dtype = cls.to_ibis(typ.value_type, typ.value_field.nullable)
            return dt.Array(value_dtype, nullable=nullable)
        elif pa.types.is_struct(typ):
            field_dtypes = {
                field.name: cls.to_ibis(field.type, field.nullable) for field in typ
            }
            return dt.Struct(field_dtypes, nullable=nullable)
        elif pa.types.is_map(typ):
            # TODO(kszucs): keys_sorted has just been exposed in pyarrow
            key_dtype = cls.to_ibis(typ.key_type, typ.key_field.nullable)
            value_dtype = cls.to_ibis(typ.item_type, typ.item_field.nullable)
            return dt.Map(key_dtype, value_dtype, nullable=nullable)
        else:
            return _from_pyarrow_types[typ](nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> pa.DataType:
        """Convert an ibis type to a pyarrow type."""

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
            short = dtype.unit.short
            if short in {"ns", "us", "ms", "s"}:
                return pa.duration(short)
            else:
                return pa.month_day_nano_interval()
        elif dtype.is_time():
            return pa.time64("ns")
        elif dtype.is_date():
            return pa.date64()
        elif dtype.is_array():
            value_field = pa.field(
                "item",
                cls.from_ibis(dtype.value_type),
                nullable=dtype.value_type.nullable,
            )
            return pa.list_(value_field)
        elif dtype.is_struct():
            fields = [
                pa.field(name, cls.from_ibis(dtype), nullable=dtype.nullable)
                for name, dtype in dtype.items()
            ]
            return pa.struct(fields)
        elif dtype.is_map():
            key_field = pa.field(
                "key",
                cls.from_ibis(dtype.key_type),
                nullable=False,  # pyarrow doesn't allow nullable keys
            )
            value_field = pa.field(
                "value",
                cls.from_ibis(dtype.value_type),
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


class PyArrowSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: Schema) -> pa.Schema:
        """Convert a schema to a pyarrow schema."""
        fields = [
            pa.field(name, PyArrowType.from_ibis(dtype), nullable=dtype.nullable)
            for name, dtype in schema.items()
        ]
        return pa.schema(fields)

    @classmethod
    def to_ibis(cls, schema: pa.Schema) -> Schema:
        """Convert a pyarrow schema to a schema."""
        fields = [(f.name, PyArrowType.to_ibis(f.type, f.nullable)) for f in schema]
        return Schema.from_tuples(fields)


class PyArrowData(DataMapper):
    @classmethod
    def infer_scalar(cls, scalar: Any) -> dt.DataType:
        """Infer the ibis type of a scalar."""
        return PyArrowType.to_ibis(pa.scalar(scalar).type)

    @classmethod
    def infer_column(cls, column: Sequence) -> dt.DataType:
        """Infer the ibis type of a sequence."""
        if isinstance(column, pa.Array):
            return PyArrowType.to_ibis(column.type)

        try:
            pyarrow_type = pa.array(column, from_pandas=True).type
            # pyarrow_type = pa.infer_type(column, from_pandas=True)
        except pa.ArrowInvalid:
            try:
                # handle embedded series objects
                return dt.highest_precedence(map(dt.infer, column))
            except TypeError:
                # we can still have a type error, e.g., float64 and string in the
                # same array
                return dt.unknown
        except pa.ArrowTypeError:
            # arrow can't infer the type
            return dt.unknown
        else:
            # arrow inferred the type, now convert that type to an ibis type
            return PyArrowType.to_ibis(pyarrow_type)

    @classmethod
    def infer_table(cls, table) -> Schema:
        """Infer the schema of a table."""
        if not isinstance(table, pa.Table):
            table = pa.table(table)

        return PyArrowSchema.to_ibis(table.schema)

    @classmethod
    def convert_scalar(cls, scalar: pa.Scalar, dtype: dt.DataType) -> pa.Scalar:
        desired_type = PyArrowType.from_ibis(dtype)
        scalar_type = scalar.type
        if scalar_type != desired_type:
            try:
                return scalar.cast(desired_type)
            except pa.ArrowNotImplementedError:
                # pyarrow doesn't support some scalar casts that are supported
                # when using arrays or tables
                return pa.array([scalar.as_py()], type=scalar_type).cast(desired_type)[
                    0
                ]
        else:
            return scalar

    @classmethod
    def convert_column(cls, column: pa.Array, dtype: dt.DataType) -> pa.Array:
        desired_type = PyArrowType.from_ibis(dtype)
        if column.type != desired_type:
            return column.cast(desired_type)
        else:
            return column

    @classmethod
    def convert_table(cls, table: pa.Table, schema: Schema) -> pa.Table:
        desired_schema = PyArrowSchema.from_ibis(schema)
        pa_schema = table.schema

        if pa_schema.names != schema.names:
            table = table.rename_columns(schema.names)

        if pa_schema != desired_schema:
            return table.cast(desired_schema)
        else:
            return table
