from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles

import ibis.backends.base.sql.alchemy.datatypes as sat
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.base.sql.alchemy.query_builder import _AlchemyTableSetFormatter
from ibis.backends.duckdb.datatypes import DuckDBType
from ibis.backends.duckdb.registry import operation_registry


class DuckDBSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _supports_tuple_syntax = True
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
    if isinstance(value_type := element.value_type, sa.types.NullType):
        # duckdb infers empty arrays with no other context as array<int32>
        typ = "INTEGER"
    else:
        typ = compiler.process(value_type, **kw)
    return f"{typ}[]"


rewrites = DuckDBSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.StringContains)
def _no_op(expr):
    return expr


class DuckDBTableSetFormatter(_AlchemyTableSetFormatter):
    def _format_sample(self, op, table):
        if op.method == "row":
            method = sa.func.bernoulli
        else:
            method = sa.func.system
        return table.tablesample(
            sampling=method(sa.literal_column(f"{op.fraction * 100} PERCENT")),
            seed=(None if op.seed is None else sa.literal_column(str(op.seed))),
        )


class DuckDBSQLCompiler(AlchemyCompiler):
    cheap_in_memory_tables = True
    translator_class = DuckDBSQLExprTranslator
    table_set_formatter_class = DuckDBTableSetFormatter
