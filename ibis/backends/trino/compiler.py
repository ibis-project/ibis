from __future__ import annotations

import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.trino.datatypes import TrinoType
from ibis.backends.trino.registry import operation_registry


class TrinoSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _integer_to_timestamp = staticmethod(sa.func.from_unixtime)

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
    _dialect_name = "trino"
    supports_unnest_in_select = False
    type_mapper = TrinoType


rewrites = TrinoSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
@rewrites(ops.StringContains)
def _no_op(expr):
    return expr


@rewrites(ops.StringContains)
def _rewrite_string_contains(op):
    return ops.GreaterEqual(ops.StringFind(op.haystack, op.needle), 0)


class TrinoSQLCompiler(AlchemyCompiler):
    cheap_in_memory_tables = False
    translator_class = TrinoSQLExprTranslator
