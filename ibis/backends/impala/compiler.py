import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.expr.operations as ops
from ibis.backends.base.sql import binary_infix_ops, operation_registry
from ibis.backends.base_sql.compiler import (
    BaseContext,
    BaseExprTranslator,
    BaseQueryBuilder,
    BaseSelectBuilder,
    BaseTableSetFormatter,
)


def _get_context():
    from ibis.backends.impala import Backend

    return Backend().dialect.make_context()


def build_ast(expr, context=None):
    if context is None:
        context = _get_context()
    builder = ImpalaQueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(expr, context):
    if context is None:
        context = _get_context()
    ast = build_ast(expr, context)
    query = ast.queries[0]

    return query


def to_sql(expr, context=None):
    if context is None:
        context = _get_context()
    query = _get_query(expr, context)
    return query.compile()


# ----------------------------------------------------------------------
# Select compilation


class ImpalaSelectBuilder(BaseSelectBuilder):
    @property
    def _select_class(self):
        return ImpalaSelect


class ImpalaQueryBuilder(BaseQueryBuilder):

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


class ImpalaTableSetFormatter(BaseTableSetFormatter):
    def _get_join_type(self, op):
        jname = self._join_names[type(op)]

        # Impala requires this
        if not op.predicates:
            jname = self._join_names[ops.CrossJoin]

        return jname


class ImpalaExprTranslator(BaseExprTranslator):
    _registry = {**operation_registry, **binary_infix_ops}
    context_class = BaseContext


rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
