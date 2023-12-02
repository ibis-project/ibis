from __future__ import annotations

import pyflink.table.types as fl

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper


class FlinkRowSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: sch.Schema | None) -> list[fl.RowType]:
        if schema is None:
            return None

        return fl.DataTypes.ROW(
            [
                fl.DataTypes.FIELD(k, FlinkType.from_ibis(v))
                for k, v in schema.fields.items()
            ]
        )


class FlinkType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: fl.DataType, nullable=True) -> dt.DataType:
        """Convert a flink type to an ibis type."""
        if typ == fl.DataTypes.STRING():
            return dt.String(nullable=nullable)
        elif typ == fl.DataTypes.BOOLEAN():
            return dt.Boolean(nullable=nullable)
        elif typ == fl.DataTypes.BYTES():
            return dt.Binary(nullable=nullable)
        elif typ == fl.DataTypes.TINYINT():
            return dt.Int8(nullable=nullable)
        elif typ == fl.DataTypes.SMALLINT():
            return dt.Int16(nullable=nullable)
        elif typ == fl.DataTypes.INT():
            return dt.Int32(nullable=nullable)
        elif typ == fl.DataTypes.BIGINT():
            return dt.Int64(nullable=nullable)
        elif typ == fl.DataTypes.FLOAT():
            return dt.Float32(nullable=nullable)
        elif typ == fl.DataTypes.DOUBLE():
            return dt.Float64(nullable=nullable)
        elif typ == fl.DataTypes.DATE():
            return dt.Date(nullable=nullable)
        elif typ == fl.DataTypes.TIME():
            return dt.Time(nullable=nullable)
        elif typ == fl.DataTypes.TIMESTAMP():
            return dt.Timestamp(nullable=nullable)
        else:
            return super().to_ibis(typ, nullable=nullable)

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> fl.DataType:
        """Convert an ibis type to a flink type."""
        if dtype.is_string():
            return fl.DataTypes.STRING()
        elif dtype.is_boolean():
            return fl.DataTypes.BOOLEAN()
        elif dtype.is_binary():
            return fl.DataTypes.BYTES()
        elif dtype.is_int8():
            return fl.DataTypes.TINYINT()
        elif dtype.is_int16():
            return fl.DataTypes.SMALLINT()
        elif dtype.is_int32():
            return fl.DataTypes.INT()
        elif dtype.is_int64():
            return fl.DataTypes.BIGINT()
        elif dtype.is_uint8():
            return fl.DataTypes.TINYINT()
        elif dtype.is_uint16():
            return fl.DataTypes.SMALLINT()
        elif dtype.is_uint32():
            return fl.DataTypes.INT()
        elif dtype.is_uint64():
            return fl.DataTypes.BIGINT()
        elif dtype.is_float16():
            return fl.DataTypes.FLOAT()
        elif dtype.is_float32():
            return fl.DataTypes.FLOAT()
        elif dtype.is_float64():
            return fl.DataTypes.DOUBLE()
        elif dtype.is_date():
            return fl.DataTypes.DATE()
        elif dtype.is_time():
            return fl.DataTypes.TIME()
        elif dtype.is_timestamp():
            return fl.DataTypes.TIMESTAMP()
        else:
            return super().from_ibis(dtype)
