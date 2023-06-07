from __future__ import annotations

from sqlalchemy.ext.compiler import compiles

import ibis.backends.base.sql.alchemy.datatypes as sat
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.duckdb.datatypes import DuckDBType
from ibis.backends.duckdb.registry import operation_registry


class DuckDBSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _dialect_name = "duckdb"
    type_mapper = DuckDBType


@compiles(sat.UInt8, "duckdb")
def compile_uint8(element, compiler, **kw):
    return "UTINYINT"


@compiles(sat.UInt16, "duckdb")
def compile_uint16(element, compiler, **kw):
    return "USMALLINT"


@compiles(sat.UInt32, "duckdb")
def compile_uint32(element, compiler, **kw):
    return "UINTEGER"


@compiles(sat.UInt64, "duckdb")
def compile_uint(element, compiler, **kw):
    return "UBIGINT"


@compiles(sat.ArrayType, "duckdb")
def compile_array(element, compiler, **kw):
    return f"{compiler.process(element.value_type, **kw)}[]"


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
