import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
)
from ibis.backends.duckdb.registry import operation_registry


class DuckDBSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    # The PostgreSQLExprTranslater maps to a `DOUBLE_PRECISION`
    # type that duckdb doesn't understand, but we probably still want
    # the updated `operation_registry` from postgres
    _type_map = AlchemyExprTranslator._type_map.copy()
    _has_reduction_filter_syntax = True


rewrites = DuckDBSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
@rewrites(ops.StringContains)
def _no_op(expr):
    return expr


class DuckDBSQLCompiler(AlchemyCompiler):
    translator_class = DuckDBSQLExprTranslator
