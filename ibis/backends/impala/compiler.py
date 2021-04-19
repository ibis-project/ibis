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


compiles = ImpalaExprTranslator.compiles
rewrites = ImpalaExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
