from sqlalchemy.dialects import postgresql

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
)

from .registry import operation_registry


class PostgresUDFNode(ops.ValueOp):
    pass


class PostgreSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {dt.Double: postgresql.DOUBLE_PRECISION, dt.Float: postgresql.REAL}
    )


rewrites = PostgreSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
def _any_all_no_op(expr):
    return expr


class PostgreSQLCompiler(AlchemyCompiler):
    translator_class = PostgreSQLExprTranslator
