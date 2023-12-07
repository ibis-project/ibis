from __future__ import annotations

from pyflink.table.types import DataType, DataTypes, RowType

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper


class FlinkRowSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: sch.Schema | None) -> list[RowType]:
        if schema is None:
            return None

        return DataTypes.ROW(
            [
                DataTypes.FIELD(k, FlinkType.from_ibis(v))
                for k, v in schema.fields.items()
            ]
        )


class FlinkType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: DataType, nullable=True) -> dt.DataType:
        """Convert a flink type to an ibis type."""
        if typ == DataTypes.STRING():
            return dt.String(nullable=nullable)
        elif typ == DataTypes.BOOLEAN():
            return dt.Boolean(nullable=nullable)
        elif typ == DataTypes.BYTES():
            return dt.Binary(nullable=nullable)
        elif typ == DataTypes.TINYINT():
            return dt.Int8(nullable=nullable)
        elif typ == DataTypes.SMALLINT():
            return dt.Int16(nullable=nullable)
        elif typ == DataTypes.INT():
            return dt.Int32(nullable=nullable)
        elif typ == DataTypes.BIGINT():
            return dt.Int64(nullable=nullable)
        elif typ == DataTypes.FLOAT():
            return dt.Float32(nullable=nullable)
        elif typ == DataTypes.DOUBLE():
            return dt.Float64(nullable=nullable)
        elif typ == DataTypes.DATE():
            return dt.Date(nullable=nullable)
        elif typ == DataTypes.TIME():
            return dt.Time(nullable=nullable)
        elif typ == DataTypes.TIMESTAMP():
            return dt.Timestamp(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> DataType:
        """Convert an ibis type to a flink type."""
        if dtype.is_string():
            return DataTypes.STRING(nullable=dtype.nullable)
        elif dtype.is_boolean():
            return DataTypes.BOOLEAN(nullable=dtype.nullable)
        elif dtype.is_binary():
            return DataTypes.BYTES(nullable=dtype.nullable)
        elif dtype.is_int8():
            return DataTypes.TINYINT(nullable=dtype.nullable)
        elif dtype.is_int16():
            return DataTypes.SMALLINT(nullable=dtype.nullable)
        elif dtype.is_int32():
            return DataTypes.INT(nullable=dtype.nullable)
        elif dtype.is_int64():
            return DataTypes.BIGINT(nullable=dtype.nullable)
        elif dtype.is_uint8():
            return DataTypes.TINYINT(nullable=dtype.nullable)
        elif dtype.is_uint16():
            return DataTypes.SMALLINT(nullable=dtype.nullable)
        elif dtype.is_uint32():
            return DataTypes.INT(nullable=dtype.nullable)
        elif dtype.is_uint64():
            return DataTypes.BIGINT(nullable=dtype.nullable)
        elif dtype.is_float16():
            return DataTypes.FLOAT(nullable=dtype.nullable)
        elif dtype.is_float32():
            return DataTypes.FLOAT(nullable=dtype.nullable)
        elif dtype.is_float64():
            return DataTypes.DOUBLE(nullable=dtype.nullable)
        elif dtype.is_date():
            return DataTypes.DATE(nullable=dtype.nullable)
        elif dtype.is_time():
            return DataTypes.TIME(nullable=dtype.nullable)
        elif dtype.is_timestamp():
            return DataTypes.TIMESTAMP(nullable=dtype.nullable)
        else:
            return super().from_ibis(dtype)

    @classmethod
    def to_string(cls, dtype: dt.DataType) -> str:
        return cls.from_ibis(dtype).type_name()
