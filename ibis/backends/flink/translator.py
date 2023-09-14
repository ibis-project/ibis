from __future__ import annotations

import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import ExprTranslator
from ibis.backends.flink.registry import operation_registry


class FlinkExprTranslator(ExprTranslator):
    _dialect_name = (
        "hive"  # TODO: neither sqlglot nor sqlalchemy supports flink dialect
    )
    _registry = operation_registry


@FlinkExprTranslator.rewrites(ops.Clip)
def _clip_no_op(op):
    return op
