from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import Compiler, ExprTranslator, TableSetFormatter
from ibis.backends.base.sql.registry import binary_infix_ops, operation_registry, unary
from ibis.expr.rewrites import rewrite_sample


class ImpalaTableSetFormatter(TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname

    def _format_in_memory_table(self, op):
        if op.data:
            return super()._format_in_memory_table(op)

        schema = op.schema
        names = schema.names
        types = schema.types
        rows = [
            f"{self._translate(ops.Cast(ops.Literal(None, dtype=dtype), to=dtype))} AS {name}"
            for name, dtype in zip(map(self._quote_identifier, names), types)
        ]
        return f"(SELECT * FROM (SELECT {', '.join(rows)}) AS _ LIMIT 0)"


class ImpalaExprTranslator(ExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops, ops.Hash: unary("fnv_hash")}
    _forbids_frame_clause = (
        *ExprTranslator._forbids_frame_clause,
        ops.Lag,
        ops.Lead,
        ops.FirstValue,
        ops.LastValue,
    )
    _unsupported_reductions = (
        ops.ApproxMedian,
        ops.ApproxCountDistinct,
        ops.GroupConcat,
    )
    _dialect_name = "hive"
    _quote_identifiers = True


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(op):
    return ops.Floor(ops.Divide(op.left, op.right))


class ImpalaCompiler(Compiler):
    translator_class = ImpalaExprTranslator
    table_set_formatter_class = ImpalaTableSetFormatter
    rewrites = Compiler.rewrites | rewrite_sample
