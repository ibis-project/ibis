import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import (
    ExprTranslator,
    QueryBuilder,
    QueryContext,
    Select,
    TableSetFormatter,
)
from ibis.backends.base.sql.registry import (
    binary_infix_ops,
    operation_registry,
)


def build_ast(expr, context=None):
    from ibis.backends.impala import Backend

    if context is None:
        context = Backend().dialect.make_context()
    return ImpalaQueryBuilder.to_ast(expr, context=context)


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


class ImpalaQueryBuilder(QueryBuilder):
    select_class = ImpalaSelect


class ImpalaTableSetFormatter(TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname


class ImpalaQueryContext(QueryContext):
    def _to_sql(self, expr, ctx):
        return ImpalaQueryBuilder.to_sql(expr, context=ctx)


class ImpalaExprTranslator(ExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops}
    context_class = ImpalaQueryContext


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
