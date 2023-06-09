from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import Compiler, ExprTranslator, TableSetFormatter
from ibis.backends.base.sql.registry import binary_infix_ops, operation_registry, unary


class ImpalaTableSetFormatter(TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname


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
