import toolz
from sqlalchemy.dialects import postgresql

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
)
from ibis.backends.base.sql.alchemy.registry import _geospatial_functions
from ibis.backends.postgres.registry import operation_registry


class PostgresUDFNode(ops.Value):
    output_shape = rlz.shape_like("args")


class PostgreSQLExprTranslator(AlchemyExprTranslator):
    _registry = toolz.merge(operation_registry, _geospatial_functions)
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _type_map = AlchemyExprTranslator._type_map.copy()
    _type_map.update(
        {
            dt.Float16: postgresql.REAL,
            dt.Float32: postgresql.REAL,
            dt.Float64: postgresql.DOUBLE_PRECISION,
        }
    )
    _has_reduction_filter_syntax = True


rewrites = PostgreSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
def _any_all_no_op(expr):
    return expr


class PostgreSQLCompiler(AlchemyCompiler):
    translator_class = PostgreSQLExprTranslator
