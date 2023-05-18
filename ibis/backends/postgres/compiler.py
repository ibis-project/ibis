from __future__ import annotations

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.postgres.datatypes import dtype_from_postgres, dtype_to_postgres
from ibis.backends.postgres.registry import operation_registry


class PostgresUDFNode(ops.Value):
    output_shape = rlz.shape_like("args")


class PostgreSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _dialect_name = "postgresql"

    # it does support it, but we can't use it because of support for pivot
    supports_unnest_in_select = False

    get_sqla_type = staticmethod(dtype_to_postgres)
    get_ibis_type = staticmethod(dtype_from_postgres)


rewrites = PostgreSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
def _any_all_no_op(expr):
    return expr


class PostgreSQLCompiler(AlchemyCompiler):
    translator_class = PostgreSQLExprTranslator
