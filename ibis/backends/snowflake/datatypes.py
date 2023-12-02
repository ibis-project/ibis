from __future__ import annotations

import sqlglot.expressions as sge

import ibis.expr.datatypes as dt
from ibis.backends.base.sqlglot.datatypes import SqlglotType


class SnowflakeType(SqlglotType):
    dialect = "snowflake"

    default_decimal_precision = 38
    default_decimal_scale = 9

    default_temporal_scale = 9

    @classmethod
    def _from_sqlglot_FLOAT(cls) -> dt.Float64:
        return dt.Float64(nullable=cls.default_nullable)

    @classmethod
    def _from_sqlglot_DECIMAL(cls, precision=None, scale=None) -> dt.Decimal:
        if scale is None or int(scale.this.this) == 0:
            return dt.Int64(nullable=cls.default_nullable)
        else:
            return super()._from_sqlglot_DECIMAL(precision, scale)

    @classmethod
    def _from_sqlglot_ARRAY(cls, value_type=None) -> dt.Array:
        assert value_type is None
        return dt.Array(dt.json, nullable=cls.default_nullable)

    @classmethod
    def _from_ibis_JSON(cls, dtype: dt.JSON) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.VARIANT)

    @classmethod
    def _from_ibis_Array(cls, dtype: dt.Array) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.ARRAY, nested=True)

    @classmethod
    def _from_ibis_Map(cls, dtype: dt.Map) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.OBJECT, nested=True)

    @classmethod
    def _from_ibis_Struct(cls, dtype: dt.Struct) -> sge.DataType:
        return sge.DataType(this=sge.DataType.Type.OBJECT, nested=True)
