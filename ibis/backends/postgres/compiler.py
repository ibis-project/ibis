from __future__ import annotations

from sqlalchemy.dialects import postgresql

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    to_sqla_type,
)
from ibis.backends.postgres.registry import operation_registry


class PostgresUDFNode(ops.Value):
    output_shape = rlz.shape_like("args")


class PostgreSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _dialect_name = "postgresql"


rewrites = PostgreSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
def _any_all_no_op(expr):
    return expr


class PostgreSQLCompiler(AlchemyCompiler):
    translator_class = PostgreSQLExprTranslator


@to_sqla_type.register(postgresql.dialect, (dt.Float16, dt.Float32))
def _float16_float32(_, itype, type_map=None):
    return postgresql.REAL


@to_sqla_type.register(postgresql.dialect, dt.Float64)
def _float64(_, itype, type_map=None):
    return postgresql.DOUBLE_PRECISION
