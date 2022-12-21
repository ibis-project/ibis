from __future__ import annotations

from trino.sqlalchemy.datatype import JSON
from trino.sqlalchemy.dialect import TrinoDialect

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.trino.registry import operation_registry


class TrinoSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _has_reduction_filter_syntax = True
    _forbids_frame_clause = (
        *AlchemyExprTranslator._forbids_frame_clause,
        ops.Lead,
        ops.Lag,
    )
    _require_order_by = (
        *AlchemyExprTranslator._require_order_by,
        ops.Lag,
        ops.Lead,
    )


rewrites = TrinoSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
@rewrites(ops.StringContains)
def _no_op(expr):
    return expr


class TrinoSQLCompiler(AlchemyCompiler):
    cheap_in_memory_tables = False
    translator_class = TrinoSQLExprTranslator


@dt.dtype.register(TrinoDialect, JSON)
def sa_jsonb(_, satype, nullable=True):
    return dt.JSON(nullable=nullable)
