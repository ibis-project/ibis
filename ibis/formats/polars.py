from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

import ibis.expr.datatypes as dt
from ibis.expr.schema import Schema
from ibis.formats import DataMapper, SchemaMapper, TableProxy, TypeMapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import pyarrow as pa


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


class PolarsType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: pl.DataType, nullable=True) -> dt.DataType:
        """Convert a polars type to an ibis type."""

        base_type = typ.base_type()
        if base_type is pl.Categorical:
            return dt.String(nullable=nullable)
        elif base_type is pl.Decimal:
            return dt.Decimal(
                precision=typ.precision, scale=typ.scale, nullable=nullable
            )
        elif base_type is pl.Datetime:
            try:
                timezone = typ.time_zone
            except AttributeError:  # pragma: no cover
                timezone = typ.tz  # pragma: no cover
            return dt.Timestamp(timezone=timezone, nullable=nullable)
        elif base_type is pl.Duration:
            try:
                time_unit = typ.time_unit
            except AttributeError:  # pragma: no cover
                time_unit = typ.tu  # pragma: no cover
            return dt.Interval(unit=time_unit, nullable=nullable)
        elif base_type is pl.List:
            return dt.Array(cls.to_ibis(typ.inner), nullable=nullable)
        elif base_type is pl.Struct:
            return dt.Struct.from_tuples(
                [(field.name, cls.to_ibis(field.dtype)) for field in typ.fields],
                nullable=nullable,
            )
        else:
            return _from_polars_types[base_type](nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> pl.DataType:
        """Convert an ibis type to a polars type."""
        if dtype.is_decimal():
            return pl.Decimal(
                precision=dtype.precision,
                scale=9 if dtype.scale is None else dtype.scale,
            )
        elif dtype.is_timestamp():
            return pl.Datetime("ns", dtype.timezone)
        elif dtype.is_interval():
            if dtype.unit.short in {"us", "ns", "ms"}:
                return pl.Duration(dtype.unit.short)
            else:
                raise ValueError(f"Unsupported polars duration unit: {dtype.unit}")
        elif dtype.is_struct():
            fields = [
                pl.Field(name=name, dtype=cls.from_ibis(dtype))
                for name, dtype in dtype.fields.items()
            ]
            return pl.Struct(fields)
        elif dtype.is_array():
            return pl.List(cls.from_ibis(dtype.value_type))
        else:
            try:
                return _to_polars_types[type(dtype)]
            except KeyError:
                raise NotImplementedError(
                    f"Converting {dtype} to polars is not supported yet"
                )


class PolarsSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: Schema) -> dict[str, pl.DataType]:
        """Convert a schema to a polars schema."""
        return {name: PolarsType.from_ibis(typ) for name, typ in schema.items()}

    @classmethod
    def to_ibis(cls, schema: dict[str, pl.DataType]) -> Schema:
        """Convert a polars schema to a schema."""
        return Schema.from_tuples(
            [(name, PolarsType.to_ibis(typ)) for name, typ in schema.items()]
        )


class PolarsData(DataMapper):
    @classmethod
    def infer_scalar(cls, scalar: Any) -> dt.DataType:
        """Infer the ibis type of a scalar."""
        return PolarsType.to_ibis(pl.Series(values=[scalar]).dtype)

    @classmethod
    def infer_column(cls, column: Sequence) -> dt.DataType:
        """Infer the ibis type of a sequence."""
        if not isinstance(column, pl.Series):
            column = pl.Series(values=column)
        return PolarsType.to_ibis(column.dtype)

    @classmethod
    def infer_table(cls, table) -> Schema:
        """Infer the schema of a table."""
        if not isinstance(table, pl.DataFrame):
            table = pl.DataFrame(table)

        return PolarsSchema.to_ibis(table.schema)

    @classmethod
    def convert_scalar(cls, df: pl.DataFrame, dtype: dt.DataType) -> Any:
        assert df.shape == (1, 1)
        df = cls.convert_table(df, Schema({df.columns[0]: dtype}))
        return df[0, 0]

    @classmethod
    def convert_column(cls, df: pl.DataFrame, dtype: dt.DataType) -> pl.Series:
        assert df.shape[1] == 1
        df = cls.convert_table(df, Schema({df.columns[0]: dtype}))
        return df[:, 0]

    @classmethod
    def convert_table(cls, df: pl.DataFrame, schema: Schema) -> pl.DataFrame:
        pl_schema = PolarsSchema.from_ibis(schema)

        if tuple(df.columns) != tuple(schema.names):
            df = df.rename(dict(zip(df.columns, schema.names)))

        if df.schema == pl_schema:
            return df
        return df.cast(pl_schema)


class PolarsDataFrameProxy(TableProxy[pl.DataFrame]):
    def to_frame(self) -> pd.DataFrame:
        return self.obj.to_pandas()

    def to_pyarrow(self, schema: Schema) -> pa.Table:
        from ibis.formats.pyarrow import PyArrowData

        table = self.obj.to_arrow()
        return PyArrowData.convert_table(table, schema)

    def to_polars(self, schema: Schema) -> pl.DataFrame:
        return self.obj
