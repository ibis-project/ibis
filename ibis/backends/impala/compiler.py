import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import (
    ExprTranslator,
    Compiler,
    QueryContext,
    Select,
    TableSetFormatter,
)
from ibis.backends.base.sql.registry import (
    binary_infix_ops,
    operation_registry,
)


class ImpalaSelect(Select):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    @property
    def translator(self):
        return ImpalaExprTranslator

    @property
    def table_set_formatter(self):
        return ImpalaTableSetFormatter


class ImpalaCompiler(Compiler):
    select_class = ImpalaSelect


class ImpalaTableSetFormatter(TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname


class ImpalaQueryContext(QueryContext):
    pass


class ImpalaExprTranslator(ExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops}
    context_class = ImpalaQueryContext


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
