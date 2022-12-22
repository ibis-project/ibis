from __future__ import annotations

from sqlalchemy.dialects import mssql

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.mssql.registry import _timestamp_from_unix, operation_registry


class MsSqlExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Boolean: mssql.BIT,
            dt.Int8: mssql.TINYINT,
            dt.Int16: mssql.SMALLINT,
            dt.Int32: mssql.INTEGER,
            dt.Int64: mssql.BIGINT,
            dt.Float16: mssql.FLOAT,
            dt.Float32: mssql.FLOAT,
            dt.Float64: mssql.REAL,
            dt.String: mssql.NVARCHAR,
        }
    )
    _bool_aggs_need_cast_to_int32 = True
    integer_to_timestamp = staticmethod(_timestamp_from_unix)
    native_json_type = False

    _forbids_frame_clause = AlchemyExprTranslator._forbids_frame_clause + (
        ops.Lag,
        ops.Lead,
    )
    _require_order_by = AlchemyExprTranslator._require_order_by + (ops.Reduction,)


rewrites = MsSqlExprTranslator.rewrites


class MsSqlCompiler(AlchemyCompiler):
    translator_class = MsSqlExprTranslator
