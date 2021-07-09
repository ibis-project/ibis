import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import (
    Compiler,
    ExprTranslator,
    TableSetFormatter,
)
from ibis.backends.base.sql.registry import (
    binary_infix_ops,
    operation_registry,
)


class ImpalaTableSetFormatter(TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname


class ImpalaExprTranslator(ExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops}


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()


class ImpalaCompiler(Compiler):
    translator_class = ImpalaExprTranslator
    table_set_formatter_class = ImpalaTableSetFormatter
