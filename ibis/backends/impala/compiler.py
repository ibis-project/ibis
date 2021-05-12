import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import (
    ExprTranslator,
    QueryBuilder,
    QueryContext,
    Select,
    SelectBuilder,
    TableSetFormatter,
)
from ibis.backends.base.sql.registry import (
    binary_infix_ops,
    operation_registry,
    quote_identifier,
)


def build_ast(expr, context=None):
    from ibis.backends.impala import Backend

    if context is None:
        context = Backend().dialect.make_context()
    builder = ImpalaQueryBuilder(expr, context=context)
    return builder.get_result()


class ImpalaTableSetFormatter(TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname

    def _quote_identifier(self, name):
        return quote_identifier(name)


class ImpalaSelect(Select):
    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    table_set_formatter = ImpalaTableSetFormatter


class ImpalaSelectBuilder(SelectBuilder):
    _select_class = ImpalaSelect


class ImpalaQueryContext(QueryContext):
    def _to_sql(self, expr, ctx):
        builder = ImpalaQueryBuilder(expr, context=ctx)
        ast = builder.get_result()
        query = ast.queries[0]
        return query.compile()


class ImpalaExprTranslator(ExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops}
    context_class = ImpalaQueryContext

    def name(self, translated, name, force=True):
        return '{} AS {}'.format(
            translated, quote_identifier(name, force=force)
        )


class ImpalaQueryBuilder(QueryBuilder):
    translator = ImpalaExprTranslator
    select_builder = ImpalaSelectBuilder


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
