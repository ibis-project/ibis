from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import Compiler, ExprTranslator, TableSetFormatter
from ibis.backends.base.sql.registry import binary_infix_ops, operation_registry


class ImpalaTableSetFormatter(TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname


class ImpalaExprTranslator(ExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops}
    _forbids_frame_clause = (
        *ExprTranslator._forbids_frame_clause,
        ops.Lag,
        ops.Lead,
        ops.FirstValue,
        ops.LastValue,
    )


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(op):
    return ops.Floor(ops.Divide(op.left, op.right))


class ImpalaCompiler(Compiler):
    translator_class = ImpalaExprTranslator
    table_set_formatter_class = ImpalaTableSetFormatter
