from __future__ import annotations

from sqlalchemy.ext.compiler import compiles

import ibis.backends.base.sql.alchemy.datatypes as sat
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    to_sqla_type,
)
from ibis.backends.duckdb.registry import operation_registry


class DuckDBSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _dialect_name = "duckdb"


@compiles(sat.UInt64, "duckdb")
@compiles(sat.UInt32, "duckdb")
@compiles(sat.UInt16, "duckdb")
@compiles(sat.UInt8, "duckdb")
def compile_uint(element, compiler, **kw):
    return element.__class__.__name__.upper()


@compiles(sat.ArrayType, "duckdb")
def compile_array(element, compiler, **kw):
    return f"{compiler.process(element.value_type, **kw)}[]"


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

    @dt.dtype.register(duckdb_engine.Dialect, sat.ArrayType)
    def _(dialect, satype, nullable=True):
        return dt.Array(dt.dtype(dialect, satype.value_type), nullable=nullable)

    @to_sqla_type.register(duckdb_engine.Dialect, dt.Array)
    def _(dialect, itype):
        return sat.ArrayType(to_sqla_type(dialect, itype.value_type))


rewrites = DuckDBSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.NotAny)
@rewrites(ops.NotAll)
@rewrites(ops.StringContains)
def _no_op(expr):
    return expr


class DuckDBSQLCompiler(AlchemyCompiler):
    cheap_in_memory_tables = True
    translator_class = DuckDBSQLExprTranslator
