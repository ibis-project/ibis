from __future__ import annotations

from sqlalchemy.ext.compiler import compiles

import ibis.backends.base.sql.alchemy.datatypes as sat
import ibis.expr.datatypes as dt
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


@compiles(sat.UInt64, "duckdb")
@compiles(sat.UInt32, "duckdb")
@compiles(sat.UInt16, "duckdb")
@compiles(sat.UInt8, "duckdb")
def compile_uint(element, compiler, **kw):
    return element.__class__.__name__.upper()


try:
    import duckdb_engine
except ImportError:
    pass
else:

    @dt.dtype.register(duckdb_engine.Dialect, sat.UInt64)
    @dt.dtype.register(duckdb_engine.Dialect, sat.UInt32)
    @dt.dtype.register(duckdb_engine.Dialect, sat.UInt16)
    @dt.dtype.register(duckdb_engine.Dialect, sat.UInt8)
    def dtype_uint(_, satype, nullable=True):
        return getattr(dt, satype.__class__.__name__)(nullable=nullable)


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
