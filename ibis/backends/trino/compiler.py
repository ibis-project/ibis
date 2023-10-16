from __future__ import annotations

import sqlalchemy as sa

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import AlchemyCompiler, AlchemyExprTranslator
from ibis.backends.base.sql.alchemy.query_builder import _AlchemyTableSetFormatter
from ibis.backends.trino.datatypes import TrinoType
from ibis.backends.trino.registry import operation_registry
from ibis.common.exceptions import UnsupportedOperationError


class TrinoSQLExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry.copy()
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = True
    _supports_tuple_syntax = True
    _integer_to_timestamp = staticmethod(sa.func.from_unixtime)

    _forbids_frame_clause = (
        *AlchemyExprTranslator._forbids_frame_clause,
        ops.Lead,
        ops.Lag,
    )
    _require_order_by = (
        *AlchemyExprTranslator._require_order_by,
        ops.Lag,
        ops.Lead,
    )
    _dialect_name = "trino"
    supports_unnest_in_select = False
    type_mapper = TrinoType


rewrites = TrinoSQLExprTranslator.rewrites


@rewrites(ops.Any)
@rewrites(ops.All)
@rewrites(ops.StringContains)
def _no_op(expr):
    return expr


@rewrites(ops.StringContains)
def _rewrite_string_contains(op):
    return ops.GreaterEqual(ops.StringFind(op.haystack, op.needle), 0)


class TrinoTableSetFormatter(_AlchemyTableSetFormatter):
    def _format_sample(self, op, table):
        if op.seed is not None:
            raise UnsupportedOperationError(
                "`Table.sample` with a random seed is unsupported"
            )
        method = sa.func.bernoulli if op.method == "row" else sa.func.system
        return table.tablesample(
            sampling=method(sa.literal_column(f"{op.fraction * 100}"))
        )

    def _format_in_memory_table(self, op, translator):
        if not op.data:
            return sa.select(
                *(
                    translator.translate(ops.Literal(None, dtype=type_)).label(name)
                    for name, type_ in op.schema.items()
                )
            ).limit(0)

        op_schema = list(op.schema.items())
        rows = [
            tuple(
                translator.translate(ops.Literal(col, dtype=type_)).label(name)
                for col, (name, type_) in zip(row, op_schema)
            )
            for row in op.data.to_frame().itertuples(index=False)
        ]
        columns = translator._schema_to_sqlalchemy_columns(op.schema)
        return sa.values(*columns, name=op.name).data(rows).select().subquery()


class TrinoSQLCompiler(AlchemyCompiler):
    cheap_in_memory_tables = False
    translator_class = TrinoSQLExprTranslator
    null_limit = sa.literal_column("ALL")
    table_set_formatter_class = TrinoTableSetFormatter
