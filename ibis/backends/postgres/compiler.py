from __future__ import annotations

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.postgres.datatypes import PostgresType
from ibis.backends.postgres.registry import operation_registry
from ibis.expr.rewrites import rewrite_sample


class PostgresUDFNode(ops.Value):
    shape = rlz.shape_like("args")


class PostgreSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _supports_tuple_syntax = True
    _dialect_name = "postgresql"

    # it does support it, but we can't use it because of support for pivot
    supports_unnest_in_select = False

    type_mapper = PostgresType


rewrites = PostgreSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
def _any_all_no_op(expr):
    return expr


class PostgreSQLCompiler(AlchemyCompiler):
    translator_class = PostgreSQLExprTranslator
    rewrites = AlchemyCompiler.rewrites | rewrite_sample
