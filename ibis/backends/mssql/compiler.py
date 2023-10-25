from __future__ import annotations

from sqlalchemy.dialects.mssql import DATETIME2

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.mssql.datatypes import MSSQLType
from ibis.backends.mssql.registry import _timestamp_from_unix, operation_registry
from ibis.expr.rewrites import rewrite_sample


class MsSqlExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _bool_aggs_need_cast_to_int32 = True

    _timestamp_type = DATETIME2
    _integer_to_timestamp = staticmethod(_timestamp_from_unix)

    native_json_type = False

    _forbids_frame_clause = AlchemyExprTranslator._forbids_frame_clause + (
        ops.Lag,
        ops.Lead,
    )
    _require_order_by = AlchemyExprTranslator._require_order_by + (ops.Reduction,)
    _dialect_name = "mssql"
    type_mapper = MSSQLType


rewrites = MsSqlExprTranslator.rewrites


class MsSqlCompiler(AlchemyCompiler):
    translator_class = MsSqlExprTranslator

    supports_indexed_grouping_keys = False
    null_limit = None
    rewrites = AlchemyCompiler.rewrites | rewrite_sample
