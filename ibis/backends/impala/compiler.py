import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.expr.operations as ops
from ibis.backends.base.sql import (
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


class ImpalaSelectBuilder(comp.SelectBuilder):
    @property
    def _select_class(self):
        return ImpalaSelect


class ImpalaQueryBuilder(comp.QueryBuilder):

    select_builder = ImpalaSelectBuilder


class ImpalaSelect(comp.Select):

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


class ImpalaTableSetFormatter(comp.TableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname

    def _quote_identifier(self, name):
        return quote_identifier(name)


class ImpalaQueryContext(comp.QueryContext):
    def _to_sql(self, expr, ctx):
        builder = ImpalaQueryBuilder(expr, context=ctx)
        ast = builder.get_result()
        query = ast.queries[0]
        return query.compile()


class ImpalaExprTranslator(comp.ExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops}
    context_class = ImpalaQueryContext

    def name(self, translated, name, force=True):
        return '{} AS {}'.format(
            translated, quote_identifier(name, force=force)
        )


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
