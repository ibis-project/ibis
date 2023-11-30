from __future__ import annotations

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


# class SnowflakeType(
#     SqlglotSnowflakeType,
# ):
#     dialect = "snowflake"
#
#     @classmethod
#     def from_ibis(cls, dtype):
#         if dtype.is_array():
#             return ARRAY
#         elif dtype.is_map() or dtype.is_struct():
#             return OBJECT
#         elif dtype.is_json():
#             return VARIANT
#         elif dtype.is_timestamp():
#             if dtype.timezone is None:
#                 return TIMESTAMP_NTZ
#             else:
#                 return TIMESTAMP_TZ
#         elif dtype.is_string():
#             # 16MB
#             return sat.VARCHAR(2**24)
#         elif dtype.is_binary():
#             # 8MB
#             return sat.VARBINARY(2**23)
#         else:
#             return super().from_ibis(dtype)
#
#     @classmethod
#     def to_ibis(cls, typ, nullable=True):
#         if isinstance(typ, (sat.REAL, sat.FLOAT, sat.Float)):
#             return dt.Float64(nullable=nullable)
#         elif isinstance(typ, TIMESTAMP_NTZ):
#             return dt.Timestamp(timezone=None, nullable=nullable)
#         elif isinstance(typ, (TIMESTAMP_LTZ, TIMESTAMP_TZ)):
#             return dt.Timestamp(timezone="UTC", nullable=nullable)
#         elif isinstance(typ, ARRAY):
#             return dt.Array(dt.json, nullable=nullable)
#         elif isinstance(typ, OBJECT):
#             return dt.Map(dt.string, dt.json, nullable=nullable)
#         elif isinstance(typ, VARIANT):
#             return dt.JSON(nullable=nullable)
#         elif isinstance(typ, sat.Numeric):
#             if (scale := typ.scale) == 0:
#                 # kind of a lie, should be int128 because 38 digits
#                 return dt.Int64(nullable=nullable)
#             else:
#                 return dt.Decimal(
#                     precision=typ.precision or 38,
#                     scale=scale or 0,
#                     nullable=nullable,
#                 )
#         else:
#             return super().to_ibis(typ, nullable=nullable)
#
#     @classmethod
#     def from_string(cls, type_string, nullable=True):
#         return SqlglotSnowflakeType.from_string(type_string, nullable=nullable)
