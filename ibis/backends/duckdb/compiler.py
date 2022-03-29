import ibis.expr.operations as ops

from ..base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from .registry import operation_registry


class DuckDBSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    # The PostgreSQLExprTranslater maps to a `DOUBLE_PRECISION`
    # type that duckdb doesn't understand, but we probably still want
    # the updated `operation_registry` from postgres
    _type_map = AlchemyExprTranslator._type_map.copy()


rewrites = DuckDBSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
def _any_all_no_op(expr):
    return expr


class DuckDBSQLCompiler(AlchemyCompiler):
    translator_class = DuckDBSQLExprTranslator
