import sqlalchemy.dialects.postgresql as pg

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyExprTranslator

from .registry import operation_registry


class PostgresUDFNode(ops.ValueOp):
    pass


def add_operation(op, translation_func):
    operation_registry[op] = translation_func


class PostgreSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update({dt.Double: pg.DOUBLE_PRECISION, dt.Float: pg.REAL})


rewrites = PostgreSQLExprTranslator.rewrites
compiles = PostgreSQLExprTranslator.compiles


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
def _any_all_no_op(expr):
    return expr
