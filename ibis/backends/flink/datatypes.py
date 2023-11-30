from __future__ import annotations

import pyflink.table.types as fl

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.formats import SchemaMapper, TypeMapper

_pyflink_types_dict = {
    dt.String: fl.DataTypes.STRING(),
    dt.Boolean: fl.DataTypes.BOOLEAN(),
    dt.Binary: fl.DataTypes.BYTES(),
    dt.Int8: fl.DataTypes.TINYINT(),
    dt.Int16: fl.DataTypes.SMALLINT(),
    dt.Int32: fl.DataTypes.INT(),
    dt.Int64: fl.DataTypes.BIGINT(),
    dt.UInt8: fl.DataTypes.TINYINT(),
    dt.UInt16: fl.DataTypes.SMALLINT(),
    dt.UInt32: fl.DataTypes.INT(),
    dt.UInt64: fl.DataTypes.BIGINT(),
    dt.Float16: fl.DataTypes.FLOAT(),
    dt.Float32: fl.DataTypes.FLOAT(),
    dt.Float64: fl.DataTypes.DOUBLE(),
    dt.Date: fl.DataTypes.DATE(),
    dt.Time: fl.DataTypes.TIME(),
    dt.Timestamp: fl.DataTypes.TIMESTAMP(),
}

_ibis_types_dict = {
    fl.DataTypes.STRING(): dt.String,
    fl.DataTypes.BOOLEAN(): dt.Boolean,
    fl.DataTypes.BYTES(): dt.Binary,
    fl.DataTypes.TINYINT(): dt.Int8,
    fl.DataTypes.SMALLINT(): dt.Int16,
    fl.DataTypes.INT(): dt.Int32,
    fl.DataTypes.BIGINT(): dt.Int64,
    fl.DataTypes.TINYINT(): dt.UInt8,
    fl.DataTypes.SMALLINT(): dt.UInt16,
    fl.DataTypes.INT(): dt.UInt32,
    fl.DataTypes.BIGINT(): dt.UInt64,
    fl.DataTypes.FLOAT(): dt.Float16,
    fl.DataTypes.FLOAT(): dt.Float32,
    fl.DataTypes.DOUBLE(): dt.Float64,
    fl.DataTypes.DATE(): dt.Date,
    fl.DataTypes.TIME(): dt.Time,
    fl.DataTypes.TIMESTAMP(): dt.Timestamp,
}


class FlinkRowSchema(SchemaMapper):
    @classmethod
    def from_ibis(cls, schema: sch.Schema) -> list[fl.RowType]:
        if schema is None:
            return None

        return fl.DataTypes.ROW(
            [
                fl.DataTypes.FIELD(key, FlinkType.from_ibis(type(value)))
                for key, value in schema.fields.items()
            ]
        )


class FlinkType(TypeMapper):
    @classmethod
    def to_ibis(cls, typ: fl.DataType, nullable=True) -> dt.DataType:
        """Convert a flink type to an ibis type."""
        return _ibis_types_dict[typ]

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> fl.DataType:
        """Convert an ibis type to a flink type."""
        return _pyflink_types_dict[dtype]
